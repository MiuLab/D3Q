import cPickle
user_actions = cPickle.load(open('user_actions.dump'))

for i in user_actions:
	info = i['inform_slots']
	request = i['request_slots']
	if len(info.keys()) > 1:
		print i

	if len(request.keys()) > 1:
		print i

	if len(info.keys()) == 1 and len(request.keys()) == 1:
		print i