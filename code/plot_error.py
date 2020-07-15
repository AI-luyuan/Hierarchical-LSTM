import matplotlib.pyplot as plt 


errors = []
for l in open('../data/yelp13/log_lstm_20', 'r', encoding='utf-8'):
	tmpdata = l.strip().split(',')
	errors.append(float(tmpdata[-1])/100.0)

errors_upa = []
for l in open('../data/yelp13/log_upa_lstm_20', 'r', encoding='utf-8'):
	tmpdata = l.strip().split(',')
	errors_upa.append(float(tmpdata[-2])/100.0)

fig = plt.figure()
plt.plot(range(1,16), errors, marker='^', label='without_upa')
plt.plot(range(1,16), errors_upa, marker='o', label='with_upa')

plt.legend()
plt.xlabel('epcho')
plt.ylabel('accuracy')
plt.title('without upa vs. with upa')

plt.show()