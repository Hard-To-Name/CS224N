import torch
import torch.nn as nn

length_k = 1000
length_mask = torch.ones(5, 5, dtype=torch.double)
batch_size = 2
for i in range(5):
    length_mask[i][i:i + length_k + 1] = 0
length_mask = torch.t(length_mask)

start_logits = torch.tensor([[1,2,3,4,5], [6,5,4,3,2]], dtype=torch.double)
end_logits = torch.tensor([[3,2,1,4,5], [4,5,6,3,2]], dtype=torch.double)

start_logits = torch.tensor([[10000, 0, 0, 0, 0], [0, 0, 10000, 0, 0]], dtype=torch.double)
end_logits = torch.tensor([[10000, 10000, 10000, 0, 0], [0, 10000, 10000, 10000, 0]], dtype=torch.double)

softmax = nn.Softmax(dim=1)
start_logits_softmax = softmax(start_logits)
end_logits_softmax = softmax(end_logits)

start_logits_softmax = torch.unsqueeze(start_logits_softmax, 2)  # (batch, query_len, 1)
end_logits_softmax = torch.unsqueeze(end_logits_softmax, 1)  # (batch, 1, query_len)

start_end_mul = torch.transpose(torch.matmul(start_logits_softmax, end_logits_softmax), dim0=1, dim1=1)
length_mul = torch.matmul(start_end_mul, length_mask)
diag = torch.diagonal(length_mul, dim1=1, dim2=2)
length_loss = torch.sum(diag) / batch_size

print('start end mul :', start_end_mul)
print('length_mask :', length_mask)
print('length_mul :', length_mul)
print('diag :', diag)

print(length_loss)