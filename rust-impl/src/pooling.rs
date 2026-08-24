/// Applies mean pooling across the sequence length dimension for each sample in the batch,
/// masking out padding tokens (where attention_mask == 0).
///
/// `last_hidden_state`: flat slice of shape `(batch_size, seq_len, hidden_dim)`
/// `attention_mask`: flat slice of shape `(batch_size, seq_len)`
/// Returns a flat vector of shape `(batch_size * hidden_dim)`.
pub fn mean_pooling(
    last_hidden_state: &[f32],
    attention_mask: &[i64],
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    let mut pooled = vec![0.0f32; batch_size * hidden_dim];

    for b in 0..batch_size {
        let mut mask_sum = 0.0f32;
        let b_offset_hidden = b * seq_len * hidden_dim;
        let b_offset_mask = b * seq_len;
        let b_offset_out = b * hidden_dim;

        for t in 0..seq_len {
            let m = attention_mask[b_offset_mask + t] as f32;
            mask_sum += m;
            let t_offset = b_offset_hidden + t * hidden_dim;

            for d in 0..hidden_dim {
                pooled[b_offset_out + d] += last_hidden_state[t_offset + d] * m;
            }
        }

        let divisor = mask_sum.max(1e-9);
        for d in 0..hidden_dim {
            pooled[b_offset_out + d] /= divisor;
        }
    }

    pooled
}

/// Applies L2 normalization across the hidden_dim vector for each sample in the batch.
/// Modifies the vector in place or returns a normalized copy.
pub fn l2_normalize_in_place(embeddings: &mut [f32], batch_size: usize, hidden_dim: usize) {
    for b in 0..batch_size {
        let offset = b * hidden_dim;
        let slice = &mut embeddings[offset..offset + hidden_dim];
        let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
        let norm = sum_sq.sqrt().max(1e-9);
        for x in slice.iter_mut() {
            *x /= norm;
        }
    }
}

/// Computes cosine similarity between two 1D float slices.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a_sq = 0.0f32;
    let mut norm_b_sq = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    let denom = (norm_a_sq.sqrt() * norm_b_sq.sqrt()).max(1e-9);
    dot / denom
}

/// Computes the maximum absolute difference between two 1D float slices.
pub fn max_absolute_difference(a: &[f32], b: &[f32]) -> f32 {
    let mut max_diff = 0.0f32;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_pooling_single() {
        // batch_size=1, seq_len=2, hidden_dim=2
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![1, 1];
        let pooled = mean_pooling(&hidden, &mask, 1, 2, 2);
        assert_eq!(pooled, vec![2.0, 3.0]);
    }

    #[test]
    fn test_mean_pooling_masked() {
        // Second token is padding (mask=0)
        let hidden = vec![1.0, 2.0, 10.0, 20.0];
        let mask = vec![1, 0];
        let pooled = mean_pooling(&hidden, &mask, 1, 2, 2);
        assert_eq!(pooled, vec![1.0, 2.0]);
    }

    #[test]
    fn test_l2_normalization() {
        let mut vec = vec![3.0, 4.0];
        l2_normalize_in_place(&mut vec, 1, 2);
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);
    }
}
