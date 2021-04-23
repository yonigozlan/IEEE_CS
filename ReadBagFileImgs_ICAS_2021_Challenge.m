!clc
clear all
close all
% Name of the bag file
bagName = 'anomaly/2020-01-17-11-38-07';
bagNameN = 'anomaly/2020-01-17-11-36-43';
fileName = '2020-01-17-11-36-43';
% open bag file
% rosbag info 'anomaly/2020-01-17-11-38-07.bag';
bag = rosbag(append(bagNameN, '.bag'));
% select topic : topics are used to save the data in ros bagfiles
bagselect2 = select(bag, 'Topic', '/mavros/imu/data');
% read messages store in the topic

msgs = readMessages(bagselect2);
% make folder to save images/vedio
folderName_curr_video = strcat(bagNameN, 'IMU');
mkdir (folderName_curr_video)
size(msgs);
msg1 = msgs(10);
showdetails(msg1{1});

figure()
hold on
ts = timeseries(bagselect2, 'Orientation.X', 'Orientation.Y', 'Orientation.Z');
plot(ts)
legend('Orientation.X', 'Orientation.Y', 'Orientation.Z')
xlswrite(append(folderName_curr_video , '/', fileName , '.xlsx'), [ts.Time, ts.Data(:,1)], 'Orientation.X' ) ;
xlswrite(append(folderName_curr_video , '/', fileName , '.xlsx'), [ts.Time, ts.Data(:,2)], 'Orientation.Y' ) ;
xlswrite(append(folderName_curr_video , '/', fileName , '.xlsx'), [ts.Time, ts.Data(:,3)], 'Orientation.Z' ) ;

figure()
hold on
ts = timeseries(bagselect2, 'AngularVelocity.X', 'AngularVelocity.Y', 'AngularVelocity.Z');
plot(ts)
legend('AngularVelocity.X', 'AngularVelocity.Y', 'AngularVelocity.Z')
xlswrite(append(folderName_curr_video , '/', fileName , '.xlsx'), [ts.Time, ts.Data(:,1)], 'AngularVelocity.X' ) ;
xlswrite(append(folderName_curr_video , '/', fileName , '.xlsx'), [ts.Time, ts.Data(:,2)], 'AngularVelocity.Y' ) ;
xlswrite(append(folderName_curr_video , '/', fileName , '.xlsx'), [ts.Time, ts.Data(:,3)], 'AngularVelocity.Z' ) ;

figure()
hold on
ts = timeseries(bagselect2, 'LinearAcceleration.X', 'LinearAcceleration.Y', 'LinearAcceleration.Z');
plot(ts)
legend('LinearAcceleration.X', 'LinearAcceleration.Y', 'LinearAcceleration.Z')

xlswrite(append(folderName_curr_video , '/', fileName , '.xlsx'), [ts.Time, ts.Data(:,1)], 'LinearAcceleration.X' ) ;
xlswrite(append(folderName_curr_video , '/', fileName , '.xlsx'), [ts.Time, ts.Data(:,2)], 'LinearAcceleration.Y' ) ;
xlswrite(append(folderName_curr_video , '/', fileName , '.xlsx'), [ts.Time, ts.Data(:,3)], 'LinearAcceleration.Z' ) ;


