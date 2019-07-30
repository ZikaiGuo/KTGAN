Codes for GAN part.

python2 cf_gan.py

Generator: gen_model.py
Discriminator: dis_model.py

Codes are adapted from IRGAN(https://github.com/geek-ai/irgan/tree/master/item_recommendation)

Need to prepare embedding file first(in .pkl format), which includes user matrix, item matrix and bias.

See cf_gan.py line 177.
param = [user_matrix, item_matrix, bias]
