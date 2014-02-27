#! /bin/sh
#ideea e ca porneste programele necesare in succesiune, cu pauze
#pentru initializari - intai socketul, apoi ping-ul si in final radioul
#sudo service apache2 restart
cd mjpg-streamer/mjpg-streamer
./mjpg_streamer -i "/usr/lib/input_uvc.so -d /dev/video0  -r 320x240 -y YUYV -f 10" -o "/usr/lib/output_http.so -p 8081 -w ./www" -o "/usr/lib/output_file.so -f /var/www/mjpg_streamer -d 10000 -c /var/www/chpict.sh" &
cd ..
cd ..

cd ~/radio
python tcp_server.py &
sleep 3
python socket_ping.py &
sleep 3
#cd ~/radio
sudo python socket_radio.py
