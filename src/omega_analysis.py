"""
Analyze which <m,p,n> cases to target for maximum research impact.

The key insight: not all rank reductions are equally valuable.
Reducing rank by 1 for a case with large marginal ω impact is worth
more than reducing rank by 1 for a case with small marginal impact.
"""

from tensor_utils import KNOWN_RANKS, compute_omega_single, get_sorted_targets
import numpy as np


def analyze_targets():
    """Print a ranked analysis of all target cases."""
    
    targets = get_sorted_targets()
    
    print("=" * 100)
    print("TARGET ANALYSIS: Rectangular Matrix Multiplication Cases")
    print("=" * 100)
    print()
    
    # Sort by a composite score: prioritize limited prior work,
    # smaller tensors, and larger marginal omega improvement
    for t in targets:
        # Score combines: tractability (smaller tensor), 
        # impact (marginal omega), and opportunity (limited prior work)
        tractability = 1.0 / np.log(t['tensor_entries'] + 1)
        impact = t['marginal_omega']
        opportunity = 2.0 if t['limited_prior_work'] else 1.0
        t['score'] = tractability * impact * opportunity
    
    targets.sort(key=lambda t: t['score'], reverse=True)
    
    print(f"{'Case':<12} {'mpn':>4} {'Tensor':>14} {'Best R':>6} "
          f"{'Std R':>5} {'Saved':>6} {'ω now':>7} {'ω if R-1':>8} "
          f"{'Δω':>7} {'Prior Work':<20} {'Score':>7}")
    print("-" * 115)
    
    for t in targets:
        case_str = f"<{t['case'][0]},{t['case'][1]},{t['case'][2]}>"
        shape_str = f"{t['tensor_shape']}"
        
        print(f"{case_str:<12} {t['mpn']:>4} {shape_str:>14} {t['best_rank']:>6} "
              f"{t['standard_rank']:>5} {t['savings_pct']:>5.1f}% "
              f"{t['omega_current']:>7.4f} {t['omega_if_improved']:>8.4f} "
              f"{t['marginal_omega']:>7.4f} "
              f"{'LIMITED' if t['limited_prior_work'] else t['source'][:18]:<20} "
              f"{t['score']:>7.4f}")
    
    print()
    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    # Top targets
    top = [t for t in targets[:10]]
    
    print("\nTop targets by composite score (tractability × impact × opportunity):\n")
    for i, t in enumerate(top):
        case_str = f"<{t['case'][0]},{t['case'][1]},{t['case'][2]}>"
        print(f"  {i+1}. {case_str} — currently rank {t['best_rank']}, "
              f"try rank {t['best_rank']-1}")
        print(f"     Tensor shape {t['tensor_shape']}, "
              f"{'limited prior work — high opportunity' if t['limited_prior_work'] else 'well-studied'}")
        print(f"     If successful: ω improves by {t['marginal_omega']:.4f}")
        print()
    
    return targets


def select_experiments(max_tensor_entries=5000, n_targets=10):
    """
    Select specific experiments to run, returning (m, p, n, target_rank) tuples.
    
    Strategy: for each promising case, try both:
    1. Matching the known best rank (validation)
    2. Beating it by 1 (the actual research contribution)
    """
    targets = get_sorted_targets()
    
    experiments = []
    for t in targets:
        if t['tensor_entries'] > max_tensor_entries:
            continue
        if len(experiments) >= n_targets * 2:
            break
        
        m, p, n = t['case']
        R = t['best_rank']
        
        # First: try to match known best (validation)
        experiments.append({
            'case': (m, p, n),
            'target_rank': R,
            'purpose': 'validate',
            'priority': t['score'],
        })
        
        # Second: try to beat it (research)
        experiments.append({
            'case': (m, p, n),
            'target_rank': R - 1,
            'purpose': 'improve',
            'priority': t['score'] * 2,  # higher priority
        })
    
    # Sort by priority
    experiments.sort(key=lambda e: e['priority'], reverse=True)
    
    return experiments


if __name__ == "__main__":
    targets = analyze_targets()