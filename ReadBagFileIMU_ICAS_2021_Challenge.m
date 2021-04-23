!clc
clear all
close all
% Name of the bag file
bagName = 'anomaly/2020-01-17-11-38-07';
% open bag file
rosbag info 'anomaly/2020-01-17-11-38-07.bag'
bag = rosbag(append(bagName, '.bag'));
% select topic : topics are used to save the data in ros bagfiles
bagselect2 = select(bag, 'Topic', '/mavros/imu/data');
% read messages store in the topic
msgs = readMessages(bagselect2);
% make folder to save images/vedio
folderName_curr_video = bagName;
mkdir (folderName_curr_video)
%  Looping over the images
for ii = 1:1:size(msgs,1)
    % To read the image
    curFrame = readImage(msgs{ii,1});
    % To flip image
    %curFrame = flip(curFrame ,1);
    % To rotate image 180 deg
    curFrame = rot90(curFrame,2);
    % To show the current image
    imshow(curFrame);
    % Get the Frame number
    curr_frame_number = num2str(ii, '%05.f');
    % To save image with GT
    imwrite(curFrame, append(folderName_curr_video , '/', curr_frame_number , '.jpeg'));
    % pause programe
    pause(0.1)
end
