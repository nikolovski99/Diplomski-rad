cfg = {
    "wlan": {
        "ssid": b"Marko's Galaxy S22",
        "pswd": b"marko999"
    },
    "myIP": "",
    
    # Test Case 3: node 0 in PC at IP '192.168.184.12' and nodes 1-2 in two rp2s with respective argvs below.
    # (Note: to setup rp2s, comment/uncomment appropriate lines, save file, and upload project to rp2.)
    #"argv": ['mp_async_example7_NN_MNIST', '3', '1', '0', '192.168.184.12'],  # argv for the node 1 in rp2 no. 1
    "argv": ['mp_async_example7_NN_MNIST', '3', '2', '0', '192.168.184.12'],  # argv for the node 2 in rp2 no. 2
}

