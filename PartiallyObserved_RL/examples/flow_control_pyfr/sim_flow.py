import numpy as np
import subprocess
import shutil
from pathlib import Path
import pandas as pd
import tempfile

class SimulatePyFR:
    """
    Fast PyFR simulator with two modes:
    - 'rollout': Fast system ID, uses flowcontrol.py plugin
    - 'feedback': Internal feedback, uses flowcontrol_feedback.py plugin
    """
    
    def __init__(self, nx, nu, dt, config_path, mesh_path, backend='cuda', init_solution=None, mode='rollout', output_dir=None, use_force_approximation=False):
        self.nx = nx
        self.nu = nu
        self.dt = dt
        self.backend = backend
        
        self.config_template = Path(config_path).resolve()
        self.mesh_path = Path(mesh_path).resolve()
        
        # Store initial solution path
        self.init_solution_path = init_solution
        
        self.mesh_basename = self.mesh_path.stem
        self.config_basename = self.config_template.stem
        
        # NEW: Mode management
        self.mode = mode
        self._setup_mode_configs()

        self.use_force_approximation = use_force_approximation

        # NEW: Output directory for saving .pyfrs files
        if output_dir is not None:
            self.output_dir = Path(output_dir).resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
        
        # Create PERSISTENT working directory
        self.work_dir = Path(tempfile.mkdtemp(prefix='pyfr_sim_'))
        
        self.current_solution = None
        self.current_time = 0.0
        
        print(f"PyFR initialized: nx={self.nx}, nu={self.nu}, dt={self.dt}")
        print(f"Mode: {self.mode}")
        print(f"Work dir: {self.work_dir}")
    
    def _setup_mode_configs(self):
        """Setup mode-specific config file paths"""
        config_dir = self.config_template.parent
        
        # Rollout config: uses flowcontrol.py
        self.config_rollout = config_dir / f'{self.config_basename}-rollout.ini'
        
        # Feedback config: uses flowcontrol_feedback.py
        self.config_feedback = config_dir / f'{self.config_basename}-feedback.ini'
        
        # Create configs if they don't exist
        if not self.config_rollout.exists():
            shutil.copy(self.config_template, self.config_rollout)
            print(f"Created rollout config: {self.config_rollout.name}")
        
        if not self.config_feedback.exists():
            # Create feedback config from template
            with open(self.config_template, 'r') as f:
                content = f.read()
            
            # Replace plugin section - USE UNDERSCORE not hyphen!
            content = content.replace(
                '[solver-plugin-flowcontrol]',
                '[solver-plugin-feedbackstep]'  # ← underscore!
            )
            
            # Add feedback-specific settings if not present
            if 'use-feedback' not in content:
                # Insert after plugin section header
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() == '[solver-plugin-feedbackstep]': #flowcontrol_feedback]
                        lines.insert(i+1, 'use-feedback = true')
                        lines.insert(i+2, 'feedback-K-file = feedback_K.npy')
                        lines.insert(i+3, 'feedback-k-file = feedback_k.npy')
                        lines.insert(i+4, 'feedback-Z-ref-file = Z_reference.npy')
                        lines.insert(i+5, 'feedback-metadata-file = feedback_metadata.npy')
                        break
                content = '\n'.join(lines)
            
            with open(self.config_feedback, 'w') as f:
                f.write(content)
            print(f"Created feedback config: {self.config_feedback.name}")
        
        # Set active config based on initial mode
        self.set_mode(self.mode)
    
    def set_mode(self, mode):
        """
        Switch between 'rollout' and 'feedback' modes.
        
        Args:
            mode: 'rollout' or 'feedback'
        """
        if mode not in ['rollout', 'feedback']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'rollout' or 'feedback'")
        
        self.mode = mode
        
        if mode == 'rollout':
            self.active_config = self.config_rollout
            print(f"[Mode] ROLLOUT → uses flowcontrol.py, reads control_sequence.npy")
        else:  # feedback
            self.active_config = self.config_feedback
            print(f"[Mode] FEEDBACK → uses flowcontrol_feedback.py, reads feedback_K.npy")
    
    def reset(self, init_solution=None):
        """Reset simulation - FIXED to preserve init_solution in work_dir"""
        
        # Track which file to preserve (if any)
        preserve_file = None
        if init_solution is not None:
            init_solution = Path(init_solution)
            # If init_solution is in work_dir, we need to preserve it temporarily
            if init_solution.parent == self.work_dir:
                import tempfile
                temp_dir = Path(tempfile.gettempdir())
                preserve_file = temp_dir / f"preserve_{init_solution.name}"
                shutil.copy(init_solution, preserve_file)
                print(f"  Preserving {init_solution.name} during reset")
        
        # Clean work directory
        for f in self.work_dir.glob('*'):
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        
        # Copy mesh and active config (mode-specific!)
        work_mesh = self.work_dir / f'{self.mesh_basename}.pyfrm'
        work_config = self.work_dir / f'{self.config_basename}.ini'
        
        shutil.copy(self.mesh_path, work_mesh)
        shutil.copy(self.active_config, work_config)  # Use mode-specific config!
        
        self.current_solution = None
        self.current_time = 0.0
        
        # Copy initial solution if provided
        if init_solution is not None:
            # Use preserved copy if we made one
            source_file = preserve_file if preserve_file is not None else init_solution
            
            self.current_solution = self.work_dir / init_solution.name
            shutil.copy(source_file, self.current_solution)
            
            # Clean up temporary file
            if preserve_file is not None and preserve_file.exists():
                preserve_file.unlink()
            
            # Extract time
            self.current_time = self._extract_time_from_solution(init_solution)
            print(f"  Starting from solution at t={self.current_time:.6f}")
    
    def _extract_time_from_solution(self, solution_file):
        """
        Extract time from solution file.
        Tries multiple methods in order of preference.
        """
        # Method 1: Try h5py if available
        try:
            import h5py
            with h5py.File(solution_file, 'r') as f:
                if 'stats' in f and 't' in f['stats'].attrs:
                    t = float(f['stats'].attrs['t'])
                    print(f"    (from HDF5 stats)")
                    return t
        except:
            pass
        
        # Method 2: Parse from filename
        try:
            # Format: inc-cylinder-re200-75.00.pyfrs -> 75.00
            time_str = solution_file.stem.split('-')[-1]
            t = float(time_str)
            print(f"    (from filename)")
            return t
        except:
            pass
        
        # Method 3: Default
        print(f"    (WARNING: Could not extract time, using 0.0)")
        return 0.0
    
    def simulate_trajectory(self, y_init=None, u=None, horizon=1):
        """
        Simulate trajectory - behavior depends on mode.
        
        Rollout mode: Uses control sequence u
        Feedback mode: Ignores u, plugin computes control from K/k
        """
        if self.mode == 'rollout':
            # Rollout mode requires control sequence
            if u is None:
                u = np.zeros((horizon, self.nu))
            else:
                u = u.reshape((horizon, self.nu))
            
            # Antisymmetric control
            u = np.column_stack([-u, u])
        
        elif self.mode == 'feedback':
            # Feedback mode: u is ignored, plugin computes control
            # But we still need to define u for internal storage
            u = np.zeros((horizon, 2))  # Placeholder
        
        # Reset simulator (uses persistent work_dir)
        self.reset(init_solution=y_init)
        
        # Storage
        self.T = np.arange(horizon + 1) * self.dt + self.current_time
        self.Y = np.zeros((horizon + 1, self.nx))
        self.U = u
        
        self.Y[0, :] = self._get_initial_observation()
        
        # Mode-specific setup
        if self.mode == 'rollout':
            self._write_control_sequence(u)
        elif self.mode == 'feedback':
            # Check that feedback files exist
            self._check_feedback_files()
        
        self._run_pyfr_trajectory(horizon)
        self._extract_observations(horizon)

        print('self.Y')
        
        return self.Y
    
    def _check_feedback_files(self):
        """Verify feedback files exist and copy to work_dir"""
        required_files = [
            'feedback_K.npy',
            'feedback_k.npy',
            'U_temp.npy',  # ← ADD THIS LINE!
            'Z_reference.npy',
            'feedback_metadata.npy'
        ]
        
        all_found = True
        for fname in required_files:
            src = Path(fname)
            if not src.exists():
                print(f"  Warning: {fname} not found - plugin will use feedforward")
                all_found = False
            else:
                # Copy to work directory
                shutil.copy(src, self.work_dir / fname)
        
        if all_found:
            print(f"  ✓ All feedback files copied to work_dir")
    
    def _get_initial_observation(self):
        """Get initial observation (zeros for now)"""
        return np.zeros(self.nx)
    
    def _write_control_sequence(self, u):
        """
        Write entire control sequence to file for plugin to read.
        Format: control_sequence.npy with shape [horizon, nu]
        """
        control_file = self.work_dir / 'control_sequence.npy'

        if len(u) == 1:
            u_extended = np.vstack([u, u])  # [u[0], u[0]]
        else:
            u_extended = u

        np.save(control_file, u)
        
        # Also write metadata
        metadata = {
            'horizon': len(u),
            'nu': self.nu,
            'dt': self.dt,
            't_start': self.current_time  # IMPORTANT: Add starting time
        }
        metadata_file = self.work_dir / 'control_metadata.npy'
        np.save(metadata_file, metadata)
        
        print(f"  Wrote control sequence: shape={u.shape}, range=[{u.min():.3f}, {u.max():.3f}]")
    
    def _run_pyfr_trajectory(self, horizon):
        """Run PyFR once for entire trajectory"""
        work_mesh = self.work_dir / f'{self.mesh_basename}.pyfrm'
        work_config = self.work_dir / f'{self.config_basename}.ini'
        
        # Calculate target end time
        tend_target = self.current_time + horizon * self.dt
        
        # Update config for full trajectory
        self._update_config(work_config, tend_target)
        
        # Build command
        if self.current_solution is None:
            cmd = ['pyfr', 'run', '-b', self.backend, str(work_mesh), str(work_config)]
        else:
            cmd = ['pyfr', 'restart', '-b', self.backend, str(work_mesh), 
                   str(self.current_solution), str(work_config)]
        
        print(f"\n[PyFR] Running from t={self.current_time:.2f} to t={tend_target:.2f}")
        print(f"  Mode: {self.mode}")
        print(f"  Command: {' '.join(cmd)}")
        
        # Run PyFR
        result = subprocess.run(
            cmd, cwd=self.work_dir, capture_output=True, text=True, timeout=3600
        )
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"PyFR failed with return code {result.returncode}")
        
        print(f"  PyFR complete!")

        # NEW: Save .pyfrs files to output directory if specified
        if self.output_dir is not None:
            self._save_solution_files()
    
    def _update_current_solution_after_step(self):
        """
        Find and update to latest .pyfrs file after PyFR run
        """
        solution_pattern = f'{self.config_basename}*.pyfrs'
        solution_files = sorted(self.work_dir.glob(solution_pattern))
        
        if solution_files:
            self.current_solution = solution_files[-1]
            self.current_time = self._extract_time_from_solution(self.current_solution)
    
    def _save_solution_files(self):
        """Copy all .pyfrs files from work_dir to output_dir"""
        if self.output_dir is None:
            return
        
        # Find all solution files
        pyfrs_files = list(self.work_dir.glob('*.pyfrs'))
        
        if not pyfrs_files:
            print("  No .pyfrs files found to save")
            return
        
        saved_count = 0
        for src in pyfrs_files:
            # Skip temporary/intermediate files if needed
            if 'tavg' in src.name or 'src' in src.name:
                continue
            
            dst = self.output_dir / src.name
            shutil.copy(src, dst)
            saved_count += 1
        
        print(f"  ✓ Saved {saved_count} solution files to {self.output_dir}")

    def _update_config(self, config_path, tend_target):
        """
        Update config for full trajectory run.
        CRITICAL: Properly handle time for restart cases.
        """
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        # For restart: PyFR reads tstart from solution file, so we MUST:
        # 1. Remove any existing tstart line
        # 2. Set tend > solution_time
        # 3. Add buffer to avoid floating point issues
        
        buffer = self.dt * 0.5#1.5  # 1.5x timestep buffer
        tend_safe = tend_target + buffer
        
        new_lines = []
        found_solver_section = False
        found_tend = False
        
        for line in lines:
            stripped = line.strip()
            
            # Track if we're in [solver-time-integrator] section
            if stripped.startswith('[solver-time-integrator]'):
                found_solver_section = True
                new_lines.append(line)
                continue
            
            # Skip tstart completely (PyFR reads from solution file)
            if stripped.startswith('tstart'):
                print(f"  Removed tstart from config (PyFR reads from solution)")
                continue
            
            # Update tend
            if stripped.startswith('tend'):
                new_lines.append(f'tend = {tend_safe:.10f}\n')
                found_tend = True
                print(f"  Updated config: tend = {tend_safe:.6f} (target={tend_target:.6f} + buffer={buffer:.6f})")
                continue
            
            # Update dt-out
            if stripped.startswith('dt-out'):
                new_lines.append(f'dt-out = {self.dt:.10f}\n')
                print(f"  Updated config: dt-out = {self.dt:.6f}")
                continue
            
            new_lines.append(line)
        
        # If tend wasn't in file, add it to solver section
        if not found_tend:
            # Find where to insert it
            for i, line in enumerate(new_lines):
                if line.strip().startswith('[solver-time-integrator]'):
                    new_lines.insert(i+1, f'tend = {tend_safe:.10f}\n')
                    print(f"  Added tend = {tend_safe:.6f} to config")
                    break
        
        with open(config_path, 'w') as f:
            f.writelines(new_lines)
        
        # Verify config is correct
        self._verify_config(config_path, tend_target)
    
    def _verify_config(self, config_path, tend_target):
        """Verify that config time is properly set"""
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check for tstart (should NOT be present for restart)
        if 'tstart' in content and self.current_solution is not None:
            print(f"  ⚠️  WARNING: tstart found in config but using restart!")
        
        # Extract tend
        for line in content.split('\n'):
            if line.strip().startswith('tend'):
                tend_actual = float(line.split('=')[1].strip())
                if tend_actual <= self.current_time:
                    raise ValueError(
                        f"Config error: tend ({tend_actual:.6f}) <= current_time ({self.current_time:.6f}). "
                        f"This will cause 'Advance time is in the past' error!"
                    )
                print(f"  ✓ Config verified: tend={tend_actual:.6f} > current_time={self.current_time:.6f}")
                break
    
    def _transform_observation(self, obs_raw):
        """
        Transform raw pressures to force coefficients.
        
        Args:
            obs_raw: array of pressure values from sensors on cylinder surface
        
        Returns:
            [CD, CL] computed via trapezoidal quadrature
        """
        if self.use_force_approximation:
            N = len(obs_raw)
            thetas = np.arange(N) * (2 * np.pi / N)  # equally spaced
            dtheta = 2 * np.pi / N
            R = 0.5
            q_inf = 0.5  # 0.5 * rho * U^2
            D = 1.0
            coeff = R * dtheta / (q_inf * D)
            
            CD = -1*coeff * np.sum(obs_raw * np.cos(thetas))
            CL = -1*coeff * np.sum(obs_raw * np.sin(thetas))
            return np.array([CD, CL])
        else:
            return obs_raw
        
    def _extract_observations(self, horizon):
        """
        Extract observations at each control timestep from saved data.
        Reads from observation log file written by plugin.
        """
        # Plugin should write observations to a file at each control timestep
        obs_log_file = self.work_dir / 'observation_log.npy'
        
        if obs_log_file.exists():
            try:
                obs_log = np.load(obs_log_file, allow_pickle=True).item()
                
                # Extract observations at each timestep
                for i in range(horizon):
                    t = self.current_time + (i + 1) * self.dt
                    
                    # Try exact match first
                    if t in obs_log:
                        # self.Y[i + 1, :] = self._transform_observation(obs_log[t])
                        if self.mode == 'feedback':
                                self.Y[i + 1, :] = obs_log[t]  # Already [CD, CL]
                        else:
                            self.Y[i + 1, :] = self._transform_observation(obs_log[t])
                    else:
                        # Try finding closest time (within tolerance)
                        times = np.array(list(obs_log.keys()))
                        closest_idx = np.argmin(np.abs(times - t))
                        closest_t = times[closest_idx]
                        
                        if abs(closest_t - t) < self.dt * 0.1:  # Within 10% of dt
                            # self.Y[i + 1, :] = self._transform_observation(obs_log[closest_t])
                            if self.mode == 'feedback':
                                self.Y[i + 1, :] = obs_log[closest_t]  # Already [CD, CL]
                            else:
                                self.Y[i + 1, :] = self._transform_observation(obs_log[closest_t])
                        else:
                            print(f"  Warning: No observation at t={t:.2f} (closest: {closest_t:.2f})")
                
                print(f"  Extracted {horizon} observations from log")
                
            except Exception as e:
                print(f"  Error reading observation log: {e}")
                self._extract_from_csv(horizon)
        else:
            print(f"  No observation log found, trying CSV...")
            self._extract_from_csv(horizon)
    
    def _extract_from_csv(self, horizon):
        """Fallback: extract from CSV output"""
        try:
            csv_file = self.work_dir / f'{self.config_basename}.csv'
            if not csv_file.exists():
                print(f"  No CSV file found")
                return
            
            df = pd.read_csv(csv_file)
            
            for i in range(horizon):
                t_target = self.current_time + (i + 1) * self.dt
                time_mask = np.abs(df.iloc[:, 0] - t_target) < self.dt * 0.1
                time_data = df[time_mask]
                
                # Extract pressure (column 3)
                pressures = time_data.iloc[:, 3].values[:self.nx]
                
                if len(pressures) >= self.nx:
                    self.Y[i + 1, :] = pressures[:self.nx]
                else:
                    self.Y[i + 1, :self.nx] = np.pad(pressures, (0, self.nx - len(pressures)))
            
            print(f"  Extracted {horizon} observations from CSV")
            
        except Exception as e:
            print(f"  Error extracting from CSV: {e}")
    
    def draw_figure(self, save_to_path=None):
        """Plot trajectory results"""
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(16, 10))
        
        # Observations over time
        plt.subplot(3, 3, 1)
        for i in range(min(self.nx, 8)):
            plt.plot(self.T, self.Y[:, i], label=f'Probe {i}', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure')
        plt.title('Probe Observations (Pressure)')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # All observations heatmap
        plt.subplot(3, 3, 2)
        plt.imshow(self.Y.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        plt.colorbar(label='Pressure')
        plt.xlabel('Time Step')
        plt.ylabel('Probe Index')
        plt.title('All Observations (Heatmap)')
        
        # Control inputs
        plt.subplot(3, 3, 3)
        plt.plot(self.T[:-1], self.U[:, 0], label='Top Jet', linewidth=2)
        plt.plot(self.T[:-1], self.U[:, 1], label='Bottom Jet', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')
        plt.title('Control Inputs (Jet Velocities)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mean observation
        plt.subplot(3, 3, 4)
        mean_obs = np.mean(self.Y, axis=1)
        plt.plot(self.T, mean_obs, '-b', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Pressure')
        plt.title('Mean Observation')
        plt.grid(True, alpha=0.3)
        
        # Std observation
        plt.subplot(3, 3, 5)
        std_obs = np.std(self.Y, axis=1)
        plt.plot(self.T, std_obs, '-r', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Std Pressure')
        plt.title('Observation Std Dev')
        plt.grid(True, alpha=0.3)
        
        # Control effort
        plt.subplot(3, 3, 6)
        control_effort = np.sum(self.U**2, axis=1)
        plt.plot(self.T[:-1], control_effort, '-k', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Effort (||u||²)')
        plt.title('Control Effort')
        plt.grid(True, alpha=0.3)
        
        # Phase plot
        if self.nx >= 2:
            plt.subplot(3, 3, 7)
            plt.plot(self.Y[:, 0], self.Y[:, 1], '-b', linewidth=1.5)
            plt.plot(self.Y[0, 0], self.Y[0, 1], 'go', markersize=10, label='Start')
            plt.plot(self.Y[-1, 0], self.Y[-1, 1], 'ro', markersize=10, label='End')
            plt.xlabel('Probe 0')
            plt.ylabel('Probe 1')
            plt.title('Phase Plot (Probe 0 vs Probe 1)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # All probe evolution
        plt.subplot(3, 3, 8)
        for i in range(min(self.nx, 5)):
            plt.plot(self.T, self.Y[:, i], label=f'Probe {i}', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure')
        plt.title('All Probe Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Control symmetry check
        plt.subplot(3, 3, 9)
        ctrl_diff = self.U[:, 0] + self.U[:, 1]
        plt.plot(self.T[:-1], ctrl_diff, '-m', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Top + Bottom Control')
        plt.title('Control Symmetry Check (should be ~0)')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'PyFR Flow Control Around 2D Cylinder (t={self.current_time:.1f} to {self.T[-1]:.1f})', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_to_path is not None:
            plt.savefig(save_to_path, format='png', dpi=150)
            print(f"\nFigure saved to: {save_to_path}")
        
        plt.show()
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
            print(f"Cleaned up work directory: {self.work_dir}")

        if self.output_dir:
            print(f"Solutions preserved in: {self.output_dir}")


def main():
    """Main function for testing"""
    from pathlib import Path
    
    pyfr_dir = Path.home() / "PyFR-Flow-Control" / "2d-inc-cylinder-base"
    
    config_path = pyfr_dir / 'inc-cylinder.ini'
    mesh_path = pyfr_dir / 'inc-cylinder.pyfrm'
    
    # Find initial solution
    pyfrs_files = sorted(pyfr_dir.glob('inc-cylinder*.pyfrs'))
    if pyfrs_files:
        init_solution = pyfrs_files[-1]
        print(f"Found {len(pyfrs_files)} solution files: {init_solution.name}")
        print("Using restart file for initial condition")
    else:
        init_solution = None
        print("No initial solution found - will start from config IC")
    
    output_dir = pyfr_dir / 'pyfr_test_results'
    output_dir.mkdir(exist_ok=True)
    init_solution = pyfr_dir / 'inc-cylinder-re200-125.00.pyfrs'# 'inc-cylinder-re100-220.00.pyfrs'##'inc-cylinder-re60-200.00.pyfrs'#'inc-cylinder-re200-75.00.pyfrs'#'inc-cylinder-re60-200.00.pyfrs'
    
    # Parameters
    state_dimension = 2
    control_dimension = 1
    dt = 0.5
    horizon = 300#30  # Number of control steps

    
    
    print('='*70)
    print('FAST PYFR FLOW CONTROL - SINGLE RUN')
    print('='*70)
    print(f'Initial solution: {Path(init_solution).name if init_solution else "None"}')
    print(f'State dimension: {state_dimension}')
    print(f'Control timestep: {dt}s')
    print(f'Horizon: {horizon} steps ({horizon*dt:.1f}s)')
    print('='*70 + '\n')
    
    # Create simulator
    import os
    cwd = os.getcwd()
    path_to_vdp = Path(cwd)/"examples/flow_control_pyfr"

    path_to_export = path_to_vdp/"Flow_Experiments/exp_re150"
    solutions_base = path_to_export / "solutions"
    solutions_base.mkdir(exist_ok=True)

    sim = SimulatePyFR(
        nx=state_dimension,
        nu=control_dimension,
        dt=dt,
        config_path=str(config_path),
        mesh_path=str(mesh_path),
        backend='cuda',
        mode='rollout',  # Start in rollout mode
        output_dir=str(solutions_base),
        use_force_approximation=True
    )
    
    # Generate control sequence
    amplitude = 0*8.0
    control = np.random.uniform(-amplitude, amplitude, horizon)#np.sin(2 * np.pi * frequency * t)
    # np.save('ctrl.npy', control)
    # iter0 = np.load('checkpoints/checkpoint_iter_3.npz')
    # Z = iter0['Z']
    # control = -20*Z[:,1,0]  # Simple proportional control on first state

    # control = np.load('U_controlled.npy')
    # umax = np.max(np.abs(control))
    # control+= umax*0.1*np.random.normal(size=control.shape)  # Add small noise

    print(f"Control shape: {control.shape}")
    print(f"Control range: [{control.min():.3f}, {control.max():.3f}]")
    
    # Run simulation
    print("\n" + "="*70)
    print("RUNNING SIMULATION")
    print("="*70)
    
    try:
        trajectory = sim.simulate_trajectory(
            y_init=init_solution,
            u=control,
            horizon=horizon
        )
        
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"  Mean: {trajectory.mean():.6f}")
        print(f"  Std:  {trajectory.std():.6f}")
        
        # Save results
        # np.save(output_dir / 'pyfr_trajectory_finCDCL.npy', trajectory)
        # np.save(output_dir / 'pyfr_control_baseline.npy', control)
        
        # Plot
        save_path = output_dir / 'pyfr_trajectory.png'
        sim.draw_figure(save_to_path=save_path)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.cleanup()


if __name__ == '__main__':
    main()