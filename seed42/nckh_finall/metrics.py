def compute_metrics(e_joules: float, total_s: float, acc: float,
                    baseline_acc: float, n_samples: int):
    """Trả về dict gồm j_per_sample, edp, e_per_b, t_per_b."""
    # J/sample
    j_per_sample = e_joules / n_samples if n_samples > 0 else None


    # Benefit (tr�nh chia 0)
    benefit = max(acc - baseline_acc, 1e-9)
    e_per_b = e_joules / benefit
    t_per_b = total_s / benefit

    return {
        "j_per_sample": j_per_sample,
        "e_per_b": e_per_b,
        "t_per_b": t_per_b,
    }