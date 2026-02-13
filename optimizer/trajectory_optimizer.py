
import math
import numpy as np
import cvxpy as cp
from scipy.interpolate import interp1d

class TrajectoryOptimizer:
    """
    Trajectory Optimizer using DL-IAPS (Dual-Loop Iterative Anchoring Path Smoothing)
    and PJSO (Piece-wise Jerk Speed Optimization).
    """

    def __init__(self, config=None):
        self.config = config if config else {}
        
        # Phase 1: Path Smoothing Parameters
        self.ps_weight_smoothness = 10.0  # Weight for smoothness
        self.ps_weight_ref_deviation = 0.0 # Weight for deviation from original path
        self.ps_max_iter = 10            # Max iterations for outer loop
        self.ps_box_margin = 0.75         # Initial safety box margin (m)
        self.ps_min_box_margin = 0.1     # Minimum safety box margin
        
        # Phase 2: Speed Profile Parameters
        self.pjso_w_ref = 1.0     # Weight for reference speed/distance
        self.pjso_w_jerk = 10.0   # Weight for jerk
        self.pjso_w_acc = 1.0     # Weight for acceleration
        self.pjso_dt = 0.2        # Time step for speed profile
        self.pjso_v_max = 5.0     # Max velocity (m/s)
        self.pjso_a_max = 2.0     # Max acceleration (m/s^2)
        self.pjso_a_min = -2.0    # Min acceleration (m/s^2)
        self.pjso_lat_acc_max = 2.0 # Max lateral acceleration (m/s^2)

    def optimize(self, path, collision_lookup):
        """
        Main entry point for optimization.
        Args:
            path: Hybrid A* Path object (xlist, ylist, yawlist, directionlist)
            collision_lookup: CollisionLookup object
        Returns:
            OptimizedTrajectory object or dict with x, y, yaw, v, a, t
        """
        print("[TrajectoryOptimizer] Start optimization...")
        
        # 1. Segment path by gear (Forward/Backward)
        segments = self._split_path_by_gear(path)
        
        opt_x, opt_y, opt_yaw, opt_v, opt_a, opt_t = [], [], [], [], [], []
        current_time = 0.0
        
        for i, segment in enumerate(segments):
            print(f"[TrajectoryOptimizer] Optimizing segment {i+1}/{len(segments)} (Direction: {'Fwd' if segment['direction'] else 'Bwd'})...")
            
            # Phase 1: Path Smoothing (DL-IAPS)
            smoothed_path = self._phase1_smooth_path(segment, collision_lookup)
            if smoothed_path is None:
                print(f"[TrajectoryOptimizer] Smoothing failed for segment {i}, using original.")
                smoothed_path = segment
            
            # Phase 2: Speed Profile (PJSO)
            speed_profile = self._phase2_speed_profile(smoothed_path)
            
            # Append to result
            # Careful with stitching: last point of segment i is first point of segment i+1
            # We skip the first point of subsequent segments to avoid duplication
            start_idx = 0 if i == 0 else 1
            
            seg_t = speed_profile['t'] + current_time
            
            opt_x.extend(speed_profile['x'][start_idx:])
            opt_y.extend(speed_profile['y'][start_idx:])
            opt_yaw.extend(speed_profile['yaw'][start_idx:])
            opt_v.extend(speed_profile['v'][start_idx:])
            opt_a.extend(speed_profile['a'][start_idx:])
            opt_t.extend(seg_t[start_idx:])
            
            current_time = seg_t[-1]

        print("[TrajectoryOptimizer] Optimization finished.")
        return {
            'x': opt_x, 'y': opt_y, 'yaw': opt_yaw,
            'v': opt_v, 'a': opt_a, 't': opt_t
        }

    def _split_path_by_gear(self, path):
        """Split path into segments where direction remains same."""
        x, y, yaw, dirs = path.xlist, path.ylist, path.yawlist, path.directionlist
        segments = []
        if not x:
            return segments
            
        curr_segment = {'x': [x[0]], 'y': [y[0]], 'yaw': [yaw[0]], 'direction': dirs[0]}
        
        for i in range(1, len(x)):
            if dirs[i] == dirs[i-1]:
                curr_segment['x'].append(x[i])
                curr_segment['y'].append(y[i])
                curr_segment['yaw'].append(yaw[i])
            else:
                # Direction changed, finish current segment
                segments.append(curr_segment)
                # Start new segment (include the switching point to maintain continuity)
                curr_segment = {'x': [x[i-1], x[i]], 'y': [y[i-1], y[i]], 'yaw': [yaw[i-1], yaw[i]], 'direction': dirs[i]}
        
        segments.append(curr_segment)
        return segments

    def _phase1_smooth_path(self, segment, collision_lookup):
        """
        Phase 1: DL-IAPS (Dual-Loop Iterative Anchoring Path Smoothing)
        """
        path_x = np.array(segment['x'])
        path_y = np.array(segment['y'])
        N = len(path_x)
        
        if N < 3:
            return segment # Too short to smooth
            
        # Initial box size
        box_size = np.full(N, self.ps_box_margin)
        
        # Iteration loop
        for iteration in range(self.ps_max_iter):
            # Inner Loop: Solve QP
            try:
                opt_x, opt_y = self._solve_smoothing_qp(path_x, path_y, box_size)
            except Exception as e:
                print(f"[Phase1] QP solver failed: {e}")
                return None
                
            # Outer Loop: Collision Check & Box Refinement
            collision_idx = []
            
            # Check deviation/collision for each point
            in_collision = False
            for k in range(N):
                # Calculate yaw roughly
                if k < N - 1:
                    dx = opt_x[k+1] - opt_x[k]
                    dy = opt_y[k+1] - opt_y[k]
                    yaw = math.atan2(dy, dx)
                    if not segment['direction']: # Reverse
                       yaw = math.atan2(-dy, -dx)
                else:
                    # Better: use previous segment yaw
                    dx = opt_x[k] - opt_x[k-1]
                    dy = opt_y[k] - opt_y[k-1]
                    yaw = math.atan2(dy, dx)
                    if not segment['direction']:
                       yaw = math.atan2(-dy, -dx)
                
                if collision_lookup.collision_detection(opt_x[k], opt_y[k], yaw):
                    if k != 0 and k != N-1: # Start/End are fixed, assume safe
                         box_size[k] *= 0.5
                         if box_size[k] < self.ps_min_box_margin:
                            # If box is too small, we might prefer to just stick with original path or best effort
                            # For robustness, we continue but mark failure if critical
                            pass
                         in_collision = True
            
            if not in_collision:
                # Success!
                # Re-calculate accurate yaw for the smoothed path
                new_yaw = []
                for k in range(N):
                    if k < N - 1:
                         dx = opt_x[k+1] - opt_x[k]
                         dy = opt_y[k+1] - opt_y[k]
                         y_ang = math.atan2(dy, dx)
                         if not segment['direction']:
                             y_ang = math.atan2(-dy, -dx)
                         new_yaw.append(y_ang)
                    else:
                         new_yaw.append(new_yaw[-1])
                
                return {
                    'x': opt_x.tolist(),
                    'y': opt_y.tolist(),
                    'yaw': new_yaw,
                    'direction': segment['direction']
                }
                
        print("[Phase1] Max iterations reached.")
        return None

    def _solve_smoothing_qp(self, ref_x, ref_y, box_size):
        """
        Construct and solve the SCP QP problem using cvxpy.
        """
        N = len(ref_x)
        P = cp.Variable((N, 2))
        
        cost = 0
        constraints = []
        
        # 1. Boundary constraints (Fixed Start & End)
        constraints += [
            P[0] == [ref_x[0], ref_y[0]],
            P[N-1] == [ref_x[N-1], ref_y[N-1]]
        ]
        
        # 2. Safety Box Constraints
        for k in range(1, N-1):
            constraints += [
                P[k, 0] >= ref_x[k] - box_size[k],
                P[k, 0] <= ref_x[k] + box_size[k],
                P[k, 1] >= ref_y[k] - box_size[k],
                P[k, 1] <= ref_y[k] + box_size[k]
            ]
            
        # 3. Objective
        for k in range(1, N-1):
            smooth_term = P[k+1] - 2*P[k] + P[k-1]
            cost += self.ps_weight_smoothness * cp.sum_squares(smooth_term)
            cost += self.ps_weight_ref_deviation * cp.sum_squares(P[k] - [ref_x[k], ref_y[k]])

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Solver status: {prob.status}")
            
        return P.value[:, 0], P.value[:, 1]


    def _phase2_speed_profile(self, path):
        """Phase 2: PJSO - Speed Profile Optimization"""
        path_x = np.array(path['x'])
        path_y = np.array(path['y'])
        
        # Calculate cumulative distance (s)
        dists = [0.0]
        for i in range(1, len(path_x)):
            d = math.sqrt((path_x[i]-path_x[i-1])**2 + (path_y[i]-path_y[i-1])**2)
            dists.append(dists[-1] + d)
        
        total_s = dists[-1]
        
        # Fallback if distance is negligible
        if total_s < 1e-3:
            return self._generate_fallback_profile(path, dists, total_s)

        # Estimate necessary time horizon T
        # T1: Time to cover distance at max speed (trapezoidal: accel -> cruise -> decel)
        # T2: Time if we never reach max speed (triangular: accel -> decel)
        est_a = self.pjso_a_max * 0.8
        est_v = self.pjso_v_max * 0.9
        
        T1 = total_s / est_v + est_v / est_a
        T2 = 2.0 * math.sqrt(total_s / est_a)
        T_horizon = max(T1, T2) * 1.5 # Add 50% buffer to be safe
        
        N = int(math.ceil(T_horizon / self.pjso_dt))
        if N < 15: N = 15 
        
        s = cp.Variable(N)
        v = cp.Variable(N)
        a = cp.Variable(N)
        
        cost = 0
        constraints = []
        
        dt = self.pjso_dt
        dt2 = dt**2
        
        # 1. Dynamics
        for k in range(N-1):
            constraints += [
                v[k+1] == v[k] + 0.5 * (a[k] + a[k+1]) * dt,
                s[k+1] == s[k] + v[k] * dt + (1.0/3.0)*a[k]*dt2 + (1.0/6.0)*a[k+1]*dt2
            ]

        # 2. Boundary Conditions
        constraints += [
            s[0] == 0.0,
            v[0] == 0.0,
            v[N-1] == 0.0,
            s[N-1] == total_s
        ]
        
        # 3. Physical Limits
        constraints += [
            v >= 0.0,
            v <= self.pjso_v_max,
            a >= self.pjso_a_min,
            a <= self.pjso_a_max
        ]
        
        # 4. Curvature Constraints
        # Simplistic approach: limit global v based on max curvature
        # More advanced: v[k] <= sqrt(a_lat / kappa(s[k])). non-convex.
        curvatures = self._calculate_curvatures(path_x, path_y)
        max_k = np.max(np.abs(curvatures)) if len(curvatures) > 0 else 0.0
        if max_k > 0.1:
            safe_v = math.sqrt(self.pjso_lat_acc_max / max_k)
            # Relaxed constraint
            constraints += [v <= max(1.5, safe_v)] 

        # 5. Objective
        for k in range(N):
            cost += self.pjso_w_acc * cp.square(a[k])
            if k < N - 1:
                jerk = (a[k+1] - a[k]) / dt
                cost += self.pjso_w_jerk * cp.square(jerk)

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception as e:
             print(f"[Phase2] Solver exception: {e}")
             return self._generate_fallback_profile(path, dists, total_s)
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
             print(f"[Phase2] Speed profile solver failed: {prob.status}")
             return self._generate_fallback_profile(path, dists, total_s)
             
        # Interpolate results
        s_val = s.value
        v_val = v.value
        a_val = a.value
        t_val = np.arange(N) * dt
        
        # Interpolate geometry from s -> x,y,yaw
        f_x = interp1d(dists, path_x, kind='linear', fill_value="extrapolate")
        f_y = interp1d(dists, path_y, kind='linear', fill_value="extrapolate")
        unwrapped_yaw = np.unwrap(path['yaw'])
        f_yaw = interp1d(dists, unwrapped_yaw, kind='linear', fill_value="extrapolate")
        
        s_val = np.clip(s_val, 0, total_s)
        new_x = f_x(s_val)
        new_y = f_y(s_val)
        new_yaw = f_yaw(s_val)
        
        final_v = v_val if path['direction'] else -v_val
        
        return {
            'x': new_x, 'y': new_y, 'yaw': new_yaw,
            'v': final_v, 'a': a_val, 't': t_val
        }

    def _generate_fallback_profile(self, path, dists, total_s):
        """Generate a simple constant speed profile as fallback."""
        print("[Phase2] Generating fallback speed profile.")
        
        dt = self.pjso_dt
        v_target = min(self.pjso_v_max, 2.0)
        acc = self.pjso_a_max * 0.5
        
        # Ramp up
        t1 = v_target / acc
        d1 = 0.5 * acc * t1**2
        
        # Ramp down
        t3 = t1
        d3 = d1
        
        d2 = total_s - d1 - d3
        
        if d2 < 0: # Triangle
             d1 = total_s / 2.0
             v_target = math.sqrt(2 * acc * d1)
             t1 = v_target / acc
             t2 = 0
             t3 = t1
             d2 = 0
        else:
             t2 = d2 / v_target
             
        total_time = t1 + t2 + t3
        N = int(math.ceil(total_time / dt)) + 2
        
        t = np.arange(N) * dt
        v = np.zeros(N)
        a = np.zeros(N)
        s = np.zeros(N)
        
        for k in range(N):
            curr_t = k * dt
            if curr_t < t1:
                current_a = acc
                current_v = current_a * curr_t
            elif curr_t < t1 + t2:
                current_a = 0
                current_v = v_target
            elif curr_t < total_time:
                current_a = -acc
                current_v = v_target - acc * (curr_t - t1 - t2)
            else:
                current_a = 0
                current_v = 0
            
            if current_v < 0: current_v = 0
            v[k] = current_v
            a[k] = current_a
            
            if k > 0:
                s[k] = s[k-1] + v[k-1] * dt
        
        # Rescale s to match end
        if s[-1] > 1e-3:
            s = s * (total_s / s[-1])
        
        # Interpolate
        f_x = interp1d(dists, path['x'], kind='linear', fill_value="extrapolate")
        f_y = interp1d(dists, path['y'], kind='linear', fill_value="extrapolate")
        unwrapped_yaw = np.unwrap(path['yaw'])
        f_yaw = interp1d(dists, unwrapped_yaw, kind='linear', fill_value="extrapolate")
        
        new_x = f_x(s)
        new_y = f_y(s)
        new_yaw = f_yaw(s)
        
        final_v = v if path['direction'] else -v
        
        return {
            'x': new_x, 'y': new_y, 'yaw': new_yaw,
            'v': final_v, 'a': a, 't': t
        }

    def _calculate_curvatures(self, bx, by):
        n = len(bx)
        k = np.zeros(n)
        for i in range(1, n - 1):
            x0, y0 = bx[i-1], by[i-1]
            x1, y1 = bx[i], by[i]
            x2, y2 = bx[i+1], by[i+1]
            dx1, dy1 = x1-x0, y1-y0
            dx2, dy2 = x2-x1, y2-y1
            try:
                area = dx1*dy2 - dy1*dx2
                len1 = math.hypot(dx1, dy1)
                len2 = math.hypot(dx2, dy2)
                len3 = math.hypot(x2-x0, y2-y0)
                if len1*len2*len3 > 1e-6:
                    k[i] = 4 * area / (len1 * len2 * len3)
            except:
                k[i] = 0
        return k
