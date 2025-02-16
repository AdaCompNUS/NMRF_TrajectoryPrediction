import os
import numpy as np
import pickle

# import csv
# file_name = './raw_data/zara03_true_pos_.csv'

# data = []
# with open(file_name, 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         data.append(row)

# transform_data = []
# for i in range(len(data[0])):
#     # frame_id, ped_id, x, y
#     line = ("%s\t%s\t%s\t%s\n" % (data[0][i], data[1][i], data[2][i], data[3][i]))
#     transform_data.append(line)

# with open('./raw_data/zara03.txt', 'w') as wf:
#     wf.writelines(transform_data)


desired_source = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
data_class = ['train', 'val', 'test']

# in <frame_id, ped_id, x, y> format

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


# Codes refer to social-stgcnn and npsn
def data_preprocess(data_dir, data_file, obs_len=8, pred_len=12, skip=1):
    '''
    Args:
    data_dirs : List of directories where raw data resides
    data_file : The file into which all the pre-processed data needs to be stored
    '''

    all_files = sorted(os.listdir(data_dir))
    all_files = [os.path.join(data_dir, _path) for _path in all_files]

    seq_len = obs_len + pred_len
    min_ped = 1
    min_history_frame = 2   # minimum for velocity calculation

    num_seqs_in_scene = []
    num_peds_in_seq = []
    seq_list = []

    for path in all_files:
        print("Now processing: ", path)
        data = read_file(path, delim='\t')

        sep = 10
        if path.split('/')[-1] == 'eth.txt': sep = 6

        frames = np.unique(data[:, 0]).tolist()

        data[:, 2] = data[:, 2] - data[:, 2].mean()
        data[:, 3] = data[:, 3] - data[:, 3].mean()

        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        
        
        # frames are not continued
        scene_srts = [0]
        for i in range(1, len(frames)): 
            if frames[i]-frames[i-1] != sep: scene_srts.append(i)
            else: continue

        # collect sequences in each scenarios
        for scene_idx in range(len(scene_srts)):
            if scene_idx == len(scene_srts) - 1:
                neighbor_frames = frames[scene_srts[scene_idx]:]
            else:
                neighbor_frames = frames[scene_srts[scene_idx]:scene_srts[scene_idx+1]]
            
            chunk_len = len(neighbor_frames)
            # sequence number in one sub-scene
            num_sequences = int(np.ceil((chunk_len - seq_len + 1) / skip))
            if num_sequences <= 0:
                continue

            valid_seqs = 0
            idx_in_frames = scene_srts[scene_idx]
            for idx in range(idx_in_frames, (idx_in_frames + num_sequences * skip + 1) - 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

                curr_seq = np.zeros((len(peds_in_curr_seq), seq_len, 2))

                num_peds_considered = 0
                for _, ped_id in enumerate(peds_in_curr_seq):
                    # current pedestrian's trajectory information: [fid, pid, x, y]
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]

                    # normalize all sequence to time [0, seq_length-1]
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    if pad_end - pad_front != seq_len:
                        continue
                    else:
                        curr_ped_pos_seq = curr_ped_seq[:, 2:]
                                    
                    # save (x,y) position sequence in pre-defined curr_seq array
                    # for pedestrian not considered, remain 0 (last x-rows)
                    _idx = num_peds_considered
                    curr_seq[_idx, :seq_len, :] = curr_ped_pos_seq[:, :2]

                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    seq_list.append(curr_seq[:num_peds_considered])
                    valid_seqs += 1

            if valid_seqs > 0:
                num_seqs_in_scene.append(valid_seqs)
        
        # print('complete one scene')
    
    seq_list = np.concatenate(seq_list, axis=0)

    # Save the arrays in the pickle file
    with open(data_file, "wb") as f:
        pickle.dump((seq_list, num_peds_in_seq, num_seqs_in_scene), f)


# pre-process for all sub-datasets
for dataset_name in desired_source:
    for split in data_class:
        data_dir = os.path.join('./raw_data', dataset_name, split)
        data_out_file = os.path.join('../processed_datasets', dataset_name + '_' + split + '.pkl')
        print("Pre-processing for: ", data_dir)
        data_preprocess(data_dir, data_out_file)

