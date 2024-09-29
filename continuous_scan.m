clear all
clc

% Define Filtering Parameters (unchanged)
min_horizontal_ang = deg2rad(-60);  % Minimum horizontal angle in radians
max_horizontal_ang = deg2rad(60);   % Maximum horizontal angle in radians
min_vertical_ang = deg2rad(-5);    % Minimum vertical angle in radians
max_vertical_ang = deg2rad(10);              % Maximum vertical angle in radians
min_dist = 5;                       % Minimum distance
max_dist = 25;                      % Maximum distance
ground_threshold = -0.5;            % Threshold for removing ground reflections (in meters)
plot_type = "both";                 % 'live' or 'filtered' or 'both' or 'none' (plotting the raw point cloud or the filtered point cloud or none)
data = "live";                      % 'live' or 'recorded'
epsilon = 0.4;                      % DBSCAN epsilon parameter
minPts = 15;                        % DBSCAN minimum points parameter

% Create a TCP/IP client
t = tcpclient('localhost', 54320);   % Connect to the Python server

% Initialize LiDAR (unchanged)
if strcmp(data, "live")
    clear lidar;
    lidar = velodynelidar('VLP16', 'Port', 2371);
    % Start LiDAR Acquisition
    start(lidar);
elseif strcmp(data, "recorded")
    [filename, pathname] = uigetfile('*.mat', 'Select the recorded LiDAR data file');
    if isequal(filename, 0)
        error('No file selected. Exiting script.');
    else
        filepath = fullfile(pathname, filename);
        load(filepath, 'frames', 'timestamps');
    end
end

% Initialize Plot (unchanged)
if strcmp(plot_type, "live") || strcmp(plot_type, "both")
    % Initialize the pcplayer viewer
    if strcmp(data, "live")
        [frame, ~] = read(lidar, 1);
        lidarViewer = pcplayer(frame.XLimits, frame.YLimits, frame.ZLimits);
    elseif strcmp(data, "recorded")
        lidarViewer = pcplayer(frames{1}.XLimits, frames{1}.YLimits, frames{1}.ZLimits);
    end
end

if strcmp(plot_type, "filtered") || strcmp(plot_type, "both")
    % Initialize the figure and scatter plot
    figure;
    h = scatter([], [], 1, 'filled');
    xlabel('X');
    ylabel('Y');
    title('2D Scatter Plot of LiDAR Points with Distance');
    colorbar;
    axis equal;
    grid on;
    
    % Set the axes and figure background color to black
    ax = gca;
    set(ax, 'Color', 'k'); % Set axes background color to black
    set(gcf, 'Color', 'k'); % Set figure background color to black
    
    % Set the color of the axes labels, title, and grid to white for visibility
    ax.XColor = 'w';
    ax.YColor = 'w';
    ax.GridColor = 'w';
    ax.Title.Color = 'w';
    ax.XLabel.Color = 'w';
    ax.YLabel.Color = 'w';
    
    % Set the color of the colorbar labels and ticks to white
    cbar = colorbar;
    cbar.Color = 'w';
end

% Main Loop to Read Frames and Update Plot (unchanged)
index = 1;
bounding_boxes = {}; % Initialize to store bounding box coordinates
while true
    if strcmp(data, "recorded")
        % Use recorded point cloud data
        if index > length(frames)
            break; % Exit loop if all recorded point clouds are processed
        end
        ptCloud = frames{index}.Location;
        index = index + 1;
    else
        % Flush the internal point cloud buffer
        flush(lidar);
        % Read the next frame
        [frame, ~] = read(lidar, 1);
        ptCloud = frame.Location;
    end
    
    x = [];
    y = [];
    z = [];
    distance = [];
    
    for i = 1:size(ptCloud, 1)
        for j = 1:size(ptCloud, 2)
            % Get the x, y, z coordinates
            x_temp = ptCloud(i, j, 1);
            y_temp = ptCloud(i, j, 2);
            z_temp = ptCloud(i, j, 3);
            
            % Skip points below the ground threshold
            if z_temp < ground_threshold
                continue;
            end
            
            % Calculate the azimuth angle (angle in the xy-plane) of the point
            azimuth = atan2(y_temp, x_temp) - 1.6;
            
            % Calculate the vertical angle (angle in the yz-plane) of the point
            vertical = atan2(z_temp, sqrt(x_temp^2 + y_temp^2));
            
            % Calculate the meridian (Euclidean) distance
            distance_temp = sqrt(x_temp^2 + y_temp^2 + z_temp^2);
            
            % Check if the point is within the specified angular and distance range
            if azimuth >= min_horizontal_ang && azimuth <= max_horizontal_ang && ...
               vertical >= min_vertical_ang && vertical <= max_vertical_ang && ...
               distance_temp >= min_dist && distance_temp <= max_dist
                
                % Store the coordinates and distance for plotting
                x = [x; x_temp];
                y = [y; y_temp];
                z = [z; z_temp];
                distance = [distance; distance_temp];
            end
        end
    end

    if strcmp(plot_type, "live") || strcmp(plot_type, "both")
        % Update the pcplayer view
        view(lidarViewer, ptCloud);
    end
    
    if strcmp(plot_type, "filtered") || strcmp(plot_type, "both")
        % Update the scatter plot
        set(h, 'XData', x, 'YData', y, 'CData', distance);
        
        % Remove previous bounding boxes
        delete(findobj(gca, 'Type', 'rectangle'));
        
        % Perform clustering on x and y coordinates
        if ~isempty(x)
            xy = [x, y];
            labels = dbscan(xy, epsilon, minPts);
            
            % Get unique labels from clustering
            unique_labels = unique(labels);
            
            % Plot 2D bounding boxes around clusters in xy plane
            current_frame_boxes = {};
            hold on;
            for k = 1:length(unique_labels)
                if unique_labels(k) == -1
                    continue; % Skip noise points
                end
                
                cluster_points = xy(labels == unique_labels(k), :);
                
                % Get bounding box limits
                x_min = min(cluster_points(:, 1));
                x_max = max(cluster_points(:, 1));
                y_min = min(cluster_points(:, 2));
                y_max = max(cluster_points(:, 2));
                
                % Store bounding box coordinates in list of lists
                box_coords = [x_min, y_min;  % Bottom-left
                              x_max, y_min;  % Bottom-right
                              x_max, y_max;  % Top-right
                              x_min, y_max]; % Top-left
                current_frame_boxes{end+1} = box_coords;
                
                % Draw bounding box (rectangle) in xy plane
                rectangle('Position', [x_min, y_min, x_max - x_min, y_max - y_min], 'EdgeColor', 'r');
            end
            % Store current frame's bounding boxes in the overall list
            bounding_boxes{end+1} = current_frame_boxes;
            hold off;
            
            % Print out coordinates of bounding boxes for this frame
            %fprintf('Coordinates of bounding boxes in frame %d:\n', index - 1);
            %for box_idx = 1:length(current_frame_boxes)
            %    fprintf('Box %d:\n', box_idx);
            %    disp(current_frame_boxes{box_idx});
            %end
            
            % Collect all bounding box coordinates into a single string
            bounding_boxes_str = '';
            for box_idx = 1:length(current_frame_boxes)
                bounding_boxes_str = [bounding_boxes_str, sprintf('%f,%f,%f,%f,%f,%f,%f,%f;', current_frame_boxes{box_idx}(:))];
            end
            % Send bounding box coordinates through the socket in one message
            if ~isempty(bounding_boxes_str)
                write(t, bounding_boxes_str);
            end
        end
        
        drawnow limitrate;
    end
    
    if strcmp(plot_type, "none")
        % Perform clustering on x and y coordinates without plotting
        if ~isempty(x)
            xy = [x, y];
            labels = dbscan(xy, epsilon, minPts);
            
            % Get unique labels from clustering
            unique_labels = unique(labels);
            
            % Collect bounding boxes around clusters in xy plane
            current_frame_boxes = {};
            for k = 1:length(unique_labels)
                if unique_labels(k) == -1
                    continue; % Skip noise points
                end
                
                cluster_points = xy(labels == unique_labels(k), :);
                
                % Get bounding box limits
                x_min = min(cluster_points(:, 1));
                x_max = max(cluster_points(:, 1));
                y_min = min(cluster_points(:, 2));
                y_max = max(cluster_points(:, 2));
                
                % Store bounding box coordinates in list of lists
                box_coords = [x_min, y_min;  % Bottom-left
                              x_max, y_min;  % Bottom-right
                              x_max, y_max;  % Top-right
                              x_min, y_max]; % Top-left
                current_frame_boxes{end+1} = box_coords;
            end
            % Store current frame's bounding boxes in the overall list
            bounding_boxes{end+1} = current_frame_boxes;
            
            % Print out coordinates of bounding boxes for this frame
            %fprintf('Coordinates of bounding boxes in frame %d:\n', index - 1);
            %for box_idx = 1:length(current_frame_boxes)
            %    fprintf('Box %d:\n', box_idx);
            %    disp(current_frame_boxes{box_idx});
            %end
            
            % Collect all bounding box coordinates into a single string
            bounding_boxes_str = '';
            for box_idx = 1:length(current_frame_boxes)
                bounding_boxes_str = [bounding_boxes_str, sprintf('%f,%f,%f,%f,%f,%f,%f,%f;', current_frame_boxes{box_idx}(:))];
            end
            % Send bounding box coordinates through the socket in one message
            if ~isempty(bounding_boxes_str)
                write(t, bounding_boxes_str);
            end
        end
        
        drawnow limitrate;
    end
    
    % Pause for a short duration to simulate live data rate
    %pause(0.1);
end

% Clear LiDAR Object (unchanged)
if strcmp(data, "live")
    clear lidar;
end

% Close the TCP client
clear t;