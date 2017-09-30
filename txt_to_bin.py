import struct

def rewrite(name):
	with open(name + '.txt', 'r') as orig, open(name + '.bin', 'wb') as dest:
		label = orig.readline()
		while label:
			data = []
			for _ in range(28):
				line = filter(lambda s: s.isdigit(), orig.readline().split(' '))
				data += list(map(int, line))
			
			assert len(data) == 784

			dest.write(struct.pack("@i784s", int(label), bytes(data)))
			label = orig.readline()


if __name__ == '__main__':
	rewrite("train")
	rewrite("test")
