import os
import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from glob import glob
import numpy as np
import skimage.io as io
from Auto_enco import MyAuto
from skimage.transform import resize
import matplotlib.pyplot as plt


# Super parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
N_TEST_IMG = 5
ROOT = '/home/disk/duli/My_autoencoder/'
DATA_ROOT = os.path.join (ROOT, 'data' + '/')
LABEL_ROOT = os.path.join(ROOT,'label' + '/')


# Read train_dataset

def get_data(sub_datafile,verbose=True):
    img = sorted(glob(os.path.join(DATA_ROOT, sub_datafile + '/','*.jpg')))
    img = np.array([resize(io.imread(frame), out_shape=(120,240), preserve_range=True) for frame in img])
    #img = img[:,45:45+120,:]
    io.imshow(img)
    if verbose:
        clip_img = np.reshape(img.transpose(3, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = img.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)
    return torch.from_numpy(clip)

#Read  train_label

def read_labels(sublabels_dir):

    lab_dir = os.path.join(LABEL_ROOT, sublabels_dir, '/', '*_train.txt')
    with open(lab_dir,'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels



def main():

    # Auto_encoder Model

    autoencoder = MyAuto()
    autoencoder.cuda()
    autoencoder.eval()


    # Show antoencoder Net structure and  its parameters
    print autoencoder
    params = list(autoencoder.parameters())
    total_num =0
    for i in params:
        layer =1
        print "The structure of this layer is:" + str(list(i.size()))
        for j in i.size():
            layer *= j
        print "The total number of parameters is:" + str(layer)
        total_num = total_num + layer
    print "Sum of the parameter total number is: " + str(total_num)

    # Get training _data

    train_data = get_data('driving_data')
    print train_data

    # Get training label
    train_label = read_labels('driving_label')
    print train_label
    # plot data & label example
    print train_data.size()
    print train_label.size()
    plt.imshow(train_data[2].numpy(), cmp='rgb')
    plt.title('%i' % train_label[2])
    plt.show()

    # Data_loader
    Loader = Data.DataLoader(Data.TensorDataset(train_data, train_label), batch_size= BATCH_SIZE, shuffle = False, num_workers=2)
#    train_Labelloader = Data.DataLoader(dataset = train_label, batch_size= BATCH_SIZE, shuffle = False, num_workers=2)


    #Optimizing parameters

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr = LR)
    loss_function = torch.nn.MSELoss()

    # Initial figure
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5,2))
    plt.ion()

    # Source data (first row ) for view
    view_data = Variable(train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255)
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28*28)), cmap = 'rgb');
        a[0][i].set_xticks(());
        a[0][i].set_yticks(())


    # Training
    for epoch in range(EPOCH):
         for step, (x,y) in enumerate (Loader):
             b_x = Variable(x.view(-1, 28*28))
             b_y = Variable(x.view(-1, 28*28))
             b_label = Variable(y)

             encoded, decoded = autoencoder(b_x)

             loss = loss_function(decoded, b_y)
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()


if __name__ == '__main__':
    main()










