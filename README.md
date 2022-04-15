First, please ensure each computing node has the python and pytorch environments. 
You can check by (Linux) : 

tec@node1:~$ python
>>>import torch
>>>exit()

If there is no error, you can continue.

#########################################################
#########################################################

How to build the DML environment:

1. We need some compute nodes (Linux servers) that can communicate with each other, it is best to be within the same LAN.

For example, here are three nodes configured as:

node1#   eno1: 11.1.1.1/255.255.255.0
node2#   eno1: 11.1.1.2/255.255.255.0
node3#   eno2: 11.1.1.3/255.255.255.0

eno1, eno2 is the NIC name in each node.

2. Choice a node as the master node, we choose node1 here. So the master ip is 11.1.1.1.

3. Edit the /train/train_dist_ddp.py

start at Line 255:

os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'  #edit the NIC name
os.environ['MASTER_ADDR'] = '11.1.1.1'  # edit the ipaddress
os.environ['MASTER_PORT'] = '23334'  # new ip port
os.environ['WORLD_SIZE'] = '3'  # edit the number of compute node
dist.init_process_group(backend, rank=rank, world_size=3)  # edit word_size = (the number of compute node)
fn(rank, size)

total_num = 3 # total_num = (the number of compute node)


4. Copy the entire "train" folder to the user directory of all nodes.

For example, we have:   (tec is my user name in each node)
node1#   /home/tec/train
node2#   /home/tec/train
node3#   /home/tec/train

5. If your NIC in each node is different, edit the code( ['GLOO_SOCKET_IFNAME'] ) in them.

For example, the NIC name in node3 is eno2, so I edit the /home/tec/train/train_dist_ddp.py in node3(11.1.1.3):

os.environ['GLOO_SOCKET_IFNAME'] = 'eno2'  #edit the NIC name

!!!!!!!!note: only edit the NIC name, the ipaddress in each node should be the same (master ip).

6. Now, we have three compute node, and each node has the same folder /home/tec/train. Or a little difference in NIC name in each code.

7. Start the DML: 

		In each node, cd the train folder

		cd /home/tec/train

Execute the following code in sequence:

In master node:
		tec@node1:~$python train_dist_ddp.py 0 $
In other node:
		tec@node2:~$python train_dist_ddp.py 1 $
		tec@node3:~$python train_dist_ddp.py 2 $

If it is going well, DML will start running.

8. The DML will iterate three times and out put each training result. The total time will output after training, and then you can ctrl+c to leave.

#########################################################
#########################################################

Some matters:

1. Line 228 in train_dist_ddp.py determine the number of iterations. You can adjust as needed, but you must ensure that the code changes on each node synchronously.

for epoch in range(3): # iteration number

2. If some errors occure during the running process, you can kill the related process to end the DML.

3. Because the goal is not to optimize the effect of DML, this code is just to open a DML job. There may be many redundant code, you can change the code according to your real environment. Once again, you must ensure that the code changes on each node synchronously.

