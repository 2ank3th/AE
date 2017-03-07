import os
import torch
from DAE import DAE
import cPickle as pickle
from torch.autograd import Variable
import csv



model = DAE()
model.load_state_dict(torch.load(os.getcwd() + "/model.pth"))


test_data = pickle.load(open("test.p", "rb"))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
i = 0
ans = []
for idx,(data, _) in enumerate(test_loader):

    data = Variable(data)
    data = data.view(-1, 784)
    output,_ = model(data, None,None, False, True)
    pred = output.data.max(1)[1]  # get the index of the max log-probability
    pred = torch.t(pred)
    for num in pred[0]:
        ans.append([i,num])
        i += 1

    with open('submission.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        for entry in ans:
            a.writerow(entry)