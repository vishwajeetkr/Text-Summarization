import json

class Vocab(object):
	def __init__(self, vocab_file_path):
		self.vocab_file_path = vocab_file_path
		
	def get_vocabulary(self, vocabulary_size):
		token_to_index, index_to_token = {}, {}
		count = 0
		vocab_file = open(self.vocab_file_path, 'r')
		for line in vocab_file:
			token = line.strip().split()[0]
			token_to_index[token] = count
			index_to_token[count] = token
			count += 1
			if count >= vocabulary_size:
				break
		return (token_to_index, index_to_token)

class Data(object):
	"""This class does the fetching and preprocessing of data.
	"""

	def __init__(self, vocab_file_path, input_file_path, max_text_length, 
				max_summary_length, vocabulary_size):
		self.input_file_path = input_file_path
		self.max_text_length = max_text_length
		self.max_summary_length = max_summary_length
		self.vocab_file_path = vocab_file_path
		self.vocabulary_size = vocabulary_size
		self.pad_token = ['PAD']
		self.vocabulary = Vocab(self.vocab_file_path)

	def pad_sequence(self, list_, max_length):
		if len(list_) < max_length:
			return list_ + self.pad_token * (max_length - len(list_))

	def pad_data(self):
		X, y = [], []
		with open(self.input_file_path, 'r') as fp:
			for line in fp:
				current_list = json.loads(line)
				for current_dict in current_list:
					text = current_dict['selftext_without_tldr_tokenized']
					summary = current_dict['trimmed_title_tokenized']
					X.append(text, self.max_text_length)
					y.append(summary, self.max_summary_length)
		return X, y

	def get_embedding_dict(self, base_embedding_dir, embedding_size):
		embeddings_index = {}
		f = open("base_embedding_dir/glove.6B." + str(embedding_size) + "d.txt", encoding='utf-8')
		for line in f:
		    values = line.split()
		    word = values[0]
		    coefs = np.asarray(values[1:], dtype='float32')
		    embeddings_index[word] = coefs
		f.close()
		return embeddings_index

	def get_contextual_embedding():
		pass