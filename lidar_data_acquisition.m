clc
clear
%% Prompt for Additional Length of Recording
prompt = {'Enter length of recording in [sec]:'};
dlgtitle = 'Recording Length';
dims = [1 50];
definput = {'10'};  % Default length of recording is 10 seconds
length = inputdlg(prompt, dlgtitle, dims, definput);

if isempty(length)
    error('No length of recording provided. Exiting script.');
else
    length_of_measurement = str2double(length{1});  % Convert the string to a double
    if isnan(length_of_measurement) || length_of_measurement <= 0
        error('Invalid length of measurement. Please enter a positive number.');
    end
end

%% Generate Default File Name with Date and Time
current_time = datetime('now', 'Format', 'yyyy-MM-dd-HHmmss');
default_filename = char(current_time);

%% Prompt for Additional File Name
prompt = {'Enter additional name for the saved file:'};
dlgtitle = 'Save File As';
dims = [1 50];
definput = {''};  % Default additional name is empty
user_input = inputdlg(prompt, dlgtitle, dims, definput);

if isempty(user_input)
    error('No file name provided. Exiting script.');
else
    user_input = user_input{1};  % Extract the string from the cell array
end

%% Create Final File Name
final_filename = [default_filename, '-', user_input, '-', length{1}, 'sec'];

%% LiDAR Data Acquisition and Saving
clear lidar
lidar = velodynelidar('VLP16', 'Port', 2371);

% Start data acquisition
start(lidar)

% Define the axis limits
xLimits = [-20, 20];
yLimits = [-20, 20];
zLimits = [-10, 10];

% Initialize the pcplayer with the specified limits
lidarViewer = pcplayer(xLimits, yLimits, zLimits);

% Read the first frame to get the limits for the pcplayer and initial timestamp
[frame, initial_timestamp] = read(lidar, 1);
timestamps = repmat(initial_timestamp, length_of_measurement * 10, 1);  % Preallocate datetime array with initial timestamp

% Read sequence of frames
numFrames = length_of_measurement * 10;
frames = cell(1, numFrames);  % Preallocate cell array for frames
frames{1} = frame;  % Store the first frame

for i = 2:numFrames
    [frame, timestamp] = read(lidar, 1);
    timestamps(i) = timestamp;  % Store the timestamp
    frames{i} = frame;  % Store the frame
    view(lidarViewer, frame);
end

% Stop data acquisition and save the data
stop(lidar)

save([final_filename, '.mat'], 'frames', 'timestamps')

% Determine fps of the recording
fps = 1 / mean(seconds(diff(timestamps)));
disp(['Frames per second: ', num2str(fps)]);
