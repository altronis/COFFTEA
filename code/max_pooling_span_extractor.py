import torch


def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.

    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```

    # Parameters

    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    # Returns

    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise RuntimeError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(
        target: torch.Tensor,
        indices: torch.LongTensor,
        flattened_indices=None,
) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.

    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.

    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](https://docs.allennlp.org/models/main/models/coref/models/coref/)
    model to select contextual word representations corresponding to the start and end indices of
    mentions.

    The key reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).

    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.

    # Returns

    selected_targets : `torch.Tensor`
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def batched_span_select(target: torch.Tensor, spans: torch.LongTensor) -> torch.Tensor:
    """
    The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.

    This function returns segmented spans in the target with respect to the provided span indices.

    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.

    # Returns

    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)

    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(
        1, 1, -1
    )
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths
    raw_span_indices = span_starts + max_span_range_indices
    # We also don't want to include span indices which greater than the sequence_length,
    # which happens because some spans near the end of the sequence
    # have a start index + max_batch_span_width > sequence_length, so we add this to the mask here.
    span_mask = span_mask & (raw_span_indices < target.size(1)) & (0 <= raw_span_indices)
    span_indices = raw_span_indices * span_mask

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = batched_index_select(target, span_indices)

    return span_embeddings, span_mask


def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min


def masked_max(
        vector: torch.Tensor,
        mask: torch.BoolTensor,
        dim: int,
        keepdim: bool = False,
) -> torch.Tensor:
    """
    To calculate max along certain dimensions on masked values

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate max, assume unmasked parts are already zeros
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate max
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the maximum values.
    """
    replaced_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


class MaxPoolingSpanExtractor:
    """
    Represents spans through the application of a dimension-wise max-pooling operation.
    Given a span x_i, ..., x_j with i,j as span_start and span_end, each dimension d
    of the resulting span s is computed via s_d = max(x_id, ..., x_jd).

    Elements masked-out by sequence_mask are ignored when max-pooling is computed.
    Span representations of masked out span_indices by span_mask are set to '0.'

    Registered as a `SpanExtractor` with name "max_pooling".

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.

    # Returns

    max_pooling_text_embeddings : `torch.FloatTensor`.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is the result of a max-pooling operation.

    """

    def __init__(self, input_dim: int):
        self._input_dim = input_dim

    def _embed_spans(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor
    ) -> torch.FloatTensor:

        if sequence_tensor.size(-1) != self._input_dim:
            raise ValueError(
                f"Dimension mismatch expected ({sequence_tensor.size(-1)}) "
                f"received ({self._input_dim})."
            )

        if sequence_tensor.shape[1] <= span_indices.max() or span_indices.min() < 0:
            raise IndexError(
                f"Span index out of range, max index ({span_indices.max()}) "
                f"or min index ({span_indices.min()}) "
                f"not valid for sequence of length ({sequence_tensor.shape[1]})."
            )

        if (span_indices[:, :, 0] > span_indices[:, :, 1]).any():
            raise IndexError(
                "Span start above span end",
            )

        # Calculate the maximum sequence length for each element in batch.
        # If span_end indices are above these length, we adjust the indices in adapted_span_indices
        # shape (batch_size), filled with the sequence length size of the sequence_tensor.
        sequence_lengths = torch.ones_like(
            sequence_tensor[:, 0, 0], dtype=torch.long
        ) * sequence_tensor.size(1)

        adapted_span_indices = torch.tensor(span_indices, device=span_indices.device)

        for b in range(sequence_lengths.shape[0]):
            adapted_span_indices[b, :, 1][adapted_span_indices[b, :, 1] >= sequence_lengths[b]] = (
                    sequence_lengths[b] - 1
            )

        # Raise Error if span indices were completely masked by sequence mask.
        # We only adjust span_end to the last valid index, so if span_end is below span_start,
        # both were above the max index:

        if (adapted_span_indices[:, :, 0] > adapted_span_indices[:, :, 1]).any():
            raise IndexError(
                "Span indices were masked out entirely by sequence mask",
            )

        # span_vals <- (batch x num_spans x max_span_length x dim)
        span_vals, span_mask = batched_span_select(sequence_tensor, adapted_span_indices)

        # The application of masked_max requires a mask of the same shape as span_vals
        # We repeat the mask along the last dimension (embedding dimension)
        repeat_dim = len(span_vals.shape) - 1
        repeat_idx = [1] * repeat_dim + [span_vals.shape[-1]]

        # ext_span_mask <- (batch x num_spans x max_span_length x dim)
        # ext_span_mask True for values in span, False for masked out values
        ext_span_mask = span_mask.unsqueeze(repeat_dim).repeat(repeat_idx)

        # max_out <- (batch x num_spans x dim)
        max_out = masked_max(span_vals, ext_span_mask, dim=-2)

        return max_out

    def forward(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor
    ):
        """
        Given a sequence tensor, extract spans, concatenate width embeddings
        when need and return representations of them.

        # Parameters

        sequence_tensor : `torch.FloatTensor`, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : `torch.LongTensor`, required.
            A tensor of shape `(batch_size, num_spans, 2)`, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the `sequence_tensor`.

        # Returns

        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
        """
        # shape (batch_size, num_spans, embedding_dim)
        span_embeddings = self._embed_spans(
            sequence_tensor, span_indices
        )
        return span_embeddings
