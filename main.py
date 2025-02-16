import argparse

import sample_mrf.sample_mrf_jrdb as mrf_jrdb
import sample_mrf.sample_mrf_nba as mrf_nba
import sample_mrf.sample_mrf_sdd as mrf_sdd
import sample_mrf.sample_mrf_ethucy as mrf_ethucy

def parse_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='nba')
	parser.add_argument('--train', action='store_true', help='Train or evaluate.')
	parser.add_argument('--log', type=str, default='', help='Use to create log file and experiment folder name.')
	parser.add_argument('--use_sampler', action='store_true', help='Whether to use the network sampler.')
	parser.add_argument('--epoch', type=int, default=150, help='Epoch to train the sampler OR evaluate.')
	return parser.parse_args()


def main(config):
	if config.dataset in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
		mrf = mrf_ethucy.Trainer(config)
	elif config.dataset == 'nba':
		mrf = mrf_nba.Trainer(config)
	elif config.dataset == 'sdd':
		mrf = mrf_sdd.Trainer(config)
	elif config.dataset == 'jrdb':
		mrf = mrf_jrdb.Trainer(config)   # jrdb dataset
	else:
		raise NotImplementedError

	if config.train == True:
		mrf.fit()
	else:
		mrf.test_model()
		# mrf.save_data()

if __name__ == "__main__":
	config = parse_config()
	main(config)
