import torch


class StaticKVCache:
    is_compileable = False

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_cache_length: int,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_cache_length = max_cache_length
        self.cache_position = 0
        self.key_cache = [None] * num_layers
        self.value_cache = [None] * num_layers

    def get_seq_length(self) -> int:
        return self.cache_position

    def update(
        self,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index {layer_index} is outside [0, {self.num_layers})"
            )

        start = self.cache_position
        end = start + key.size(-2)
        if end > self.max_cache_length:
            raise ValueError(
                f"KV cache length {end} exceeds static capacity "
                f"{self.max_cache_length}"
            )

        if self.key_cache[layer_index] is None:
            cache_shape = (
                self.batch_size,
                key.size(1),
                self.max_cache_length,
                key.size(3),
            )
            self.key_cache[layer_index] = key.new_empty(cache_shape)
            self.value_cache[layer_index] = value.new_empty(cache_shape)

        key_buffer = self.key_cache[layer_index]
        value_buffer = self.value_cache[layer_index]
        key_buffer[:, :, start:end, :].copy_(key)
        value_buffer[:, :, start:end, :].copy_(value)

        if layer_index == self.num_layers - 1:
            self.cache_position = end

        return key_buffer[:, :, :end, :], value_buffer[:, :, :end, :]

    def reset(self):
        self.cache_position = 0
