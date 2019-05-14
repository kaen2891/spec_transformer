# spec_transformer

git hub upload test


# Dataset Input Information
Mel_spec: (n_mel, second) 128, 500 -> 5seconds

Then we divide time step by 100
==> 500 -> 5

So in this, we will use autoencoder or mean 100 to 1 dim

Finally we get mel_spec: (128, n)

n is time step(second)

So, transpose this ==> (n, 128) and cut by n

Finally, we got 1 x 128 speech embedding

# Transformer Input
Transformer Encoder Input dim: batch x n x 128
Transformer Positional Encoding dim: batch x n x 128

Query, Key, Value = 128 / 8 ==> 16 dim.

Divide by square(Query_dim) ==> 4 dim.




