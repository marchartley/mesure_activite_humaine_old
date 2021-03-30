import os

accessed_servers = {}
def grant_access_server(hostname = None, share_folder_name = None, user = None, password = None, verbose = False):
	import pywintypes
	import win32wnet
	if user is None:
		user = 'mhartley'
	if password is None:
		password = 'achanger'
	if hostname is None:
		hostname = "lst-data2.local.univ-savoie.fr"
	if share_folder_name is None:
		share_folder_name = 'iot'
	SHARE_FULL_NAME = os.path.sep * 2 + os.path.sep.join((hostname, share_folder_name))

	if SHARE_FULL_NAME in accessed_servers and user in accessed_servers[SHARE_FULL_NAME]:
		if verbose:
			print(f"Already connected to {SHARE_FULL_NAME}")
		return True

	net_resource = win32wnet.NETRESOURCE()
	net_resource.lpRemoteName = SHARE_FULL_NAME
	CONNECT_INTERACTIVE = 0x00000008
	flags = 0
	# flags |= CONNECT_INTERACTIVE
	if verbose:
		print("Trying to create connection to: {:s}".format(SHARE_FULL_NAME))
	try:
		win32wnet.WNetAddConnection2(net_resource, password, user, flags)
	except pywintypes.error as e:
		print(e)
		return False
	else:
		if verbose:
			print("Success!")
		if SHARE_FULL_NAME not in accessed_servers:
			accessed_servers[SHARE_FULL_NAME] = []
		if user not in accessed_servers[SHARE_FULL_NAME]:
			accessed_servers[SHARE_FULL_NAME].append(user)
		return True