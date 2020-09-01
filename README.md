# quartet_loss
Implementation of the Quartet loss from the paper: "Optimizing Neural Network Embeddings Using a Pair-Wise Loss for Text-Independent Speaker Verification"

pdf: 
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9003794&casa_token=2iZlpXu8J8IAAAAA:hPuZ6x0ms8PlkUU-uC1nBpp1Fyuvgfaw2E3BOARVg52d8KNSpd64o4ftYhn0ij9VJIjEMgiGnw&tag=1

citation: 
Dhamyal, Hira, et al. "Optimizing Neural Network Embeddings Using a Pair-Wise Loss for Text-Independent Speaker Verification." 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2019.


    ## The general structure of the code should be: 
    '''
    for epoch in range(NO_EPOCH):
        scheduler.step()
        data_list, N = trainset.get_batch()
        while (len(data_list) == batchsize and trainset.spk_done < trainset.total_same_spk):
            for i, data in enumerate(dataloader):
                optimizer.zero_grad()
                # torch.FloatTensor
                data = Variable(data.cuda(), requires_grad=True)
                feature = net(data)
                loss = criterion.forward(feature, batchsize)
                loss.backward()
                optimizer.step()
                data_list, N = trainset.get_batch()
        # reset variables for trainset
        trainset.reset()
    '''
