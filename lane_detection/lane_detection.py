import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

class Camera:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
    
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # radius of curvature of the line in meters
        self.radius_of_curvature = None 
        # distance in meters of vehicle center from the line
        self.line_base_pos = None 
        # x values for detected line pixels
        self.allx = None  
        # y values for detected line pixels
        self.ally = None
        # smoothing - last n fits
        self.recent_fits = []
        self.n_fits = 5  # Store last 5 valid fits for smoothing
        self.best_fit = None  # Average of recent fits
        self.miss_count = 0  # Counter for consecutive missed detections

def color_threshold(img, s_thresh=(170, 255), b_thresh=(155, 200)):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:,:,2]
    
    # Also try using the L channel from HLS for better white line detection
    # (especially useful in bright conditions)
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 210) & (l_channel <= 255)] = 1
    
    # Threshold S channel (yellow lines)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Threshold B channel (yellow lines in different lighting)
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
    
    # Combine the three binary thresholds
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (b_binary == 1) | (l_binary == 1)] = 1
    
    return combined_binary

def perspective_transform(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[190, 720], [582, 457], [701, 457], [1145, 720]])
    offset = [150, 0]
    dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset, 
                      np.array([src[3, 0], 0]) - offset, src[3] - offset])
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # Return the resulting image and matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, M, Minv

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int32(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped, leftx, lefty, rightx, righty):
    # Generate y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    # Fit a second order polynomial to each using `np.polyfit`
    try:
        # Check if we have enough points for a reliable fit (minimum 10 points)
        if len(leftx) > 10 and len(rightx) > 10:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        
            # Generate x values for plotting
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            return left_fit, right_fit, left_fitx, right_fitx, ploty
        else:
            return None, None, None, None, ploty
    except:
        return None, None, None, None, ploty

def search_around_poly(binary_warped, left_fit, right_fit, margin=100):
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set the area of search based on activated x-values
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty

def calculate_curvature(left_fitx, right_fitx, ploty, img_width):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    y_eval = np.max(ploty)
    
    # Calculate radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Calculate vehicle position
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    vehicle_offset = (img_width/2 - lane_center) * xm_per_pix
    
    return left_curverad, right_curverad, vehicle_offset

def draw_lane(original_img, binary_warped, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    
    return result

def process_frame(image, left_line, right_line, camera):
    # Undistort image
    undist = camera.undistort(image)
    
    # Color and gradient threshold
    binary = color_threshold(undist)
    
    # Perspective transform
    binary_warped, M, Minv = perspective_transform(binary)
    
    # If lines were previously detected, search around the polynomial
    if left_line.detected and right_line.detected:
        leftx, lefty, rightx, righty = search_around_poly(binary_warped, left_line.current_fit, right_line.current_fit)
    else:
        # Find lane pixels using sliding window
        leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_warped)
    
    # Fit polynomial
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(binary_warped, leftx, lefty, rightx, righty)
    
    # Check if fit was successful
    if left_fit is not None and right_fit is not None:
        # Validate the detected lanes (sanity check)
        lane_width = np.mean(right_fitx - left_fitx)
        
        # Basic sanity check: lane width should be reasonable
        if 500 < lane_width < 900:  # Reasonable lane width in pixels
            # Update line properties
            left_line.detected = True
            right_line.detected = True
            left_line.current_fit = left_fit
            right_line.current_fit = right_fit
            left_line.miss_count = 0
            right_line.miss_count = 0
            
            # Add to recent fits (for smoothing)
            left_line.recent_fits.append(left_fit)
            right_line.recent_fits.append(right_fit)
            
            # Keep only the last n_fits
            if len(left_line.recent_fits) > left_line.n_fits:
                left_line.recent_fits.pop(0)
                right_line.recent_fits.pop(0)
            
            # Calculate average fit for smoothing
            left_line.best_fit = np.mean(left_line.recent_fits, axis=0)
            right_line.best_fit = np.mean(right_line.recent_fits, axis=0)
            
            # Use smoothed fit for visualization
            smooth_left_fitx = left_line.best_fit[0]*ploty**2 + left_line.best_fit[1]*ploty + left_line.best_fit[2]
            smooth_right_fitx = right_line.best_fit[0]*ploty**2 + right_line.best_fit[1]*ploty + right_line.best_fit[2]
            
            # Calculate curvature
            left_curverad, right_curverad, vehicle_offset = calculate_curvature(smooth_left_fitx, smooth_right_fitx, ploty, binary_warped.shape[1])
            
            # Draw lane
            result = draw_lane(undist, binary_warped, smooth_left_fitx, smooth_right_fitx, ploty, Minv)
        else:
            # Invalid lane width, use previous fit if available
            left_line.miss_count += 1
            right_line.miss_count += 1
            
            if left_line.best_fit is not None and len(left_line.recent_fits) > 0:
                # Use the last good fit
                smooth_left_fitx = left_line.best_fit[0]*ploty**2 + left_line.best_fit[1]*ploty + left_line.best_fit[2]
                smooth_right_fitx = right_line.best_fit[0]*ploty**2 + right_line.best_fit[1]*ploty + right_line.best_fit[2]
                
                # Calculate curvature
                left_curverad, right_curverad, vehicle_offset = calculate_curvature(smooth_left_fitx, smooth_right_fitx, ploty, binary_warped.shape[1])
                
                # Draw lane
                result = draw_lane(undist, binary_warped, smooth_left_fitx, smooth_right_fitx, ploty, Minv)
            else:
                # No previous valid fits available
                left_line.detected = False
                right_line.detected = False
                result = undist
    else:
        # If fit failed, try to use previous fit
        left_line.miss_count += 1
        right_line.miss_count += 1
        
        # If we have previous good fits and haven't missed too many frames
        if left_line.best_fit is not None and right_line.best_fit is not None and left_line.miss_count < 10:
            # Use the last good fit
            smooth_left_fitx = left_line.best_fit[0]*ploty**2 + left_line.best_fit[1]*ploty + left_line.best_fit[2]
            smooth_right_fitx = right_line.best_fit[0]*ploty**2 + right_line.best_fit[1]*ploty + right_line.best_fit[2]
            
            # Calculate curvature using previous good fit
            left_curverad, right_curverad, vehicle_offset = calculate_curvature(smooth_left_fitx, smooth_right_fitx, ploty, binary_warped.shape[1])
            
            # Draw lane
            result = draw_lane(undist, binary_warped, smooth_left_fitx, smooth_right_fitx, ploty, Minv)
        else:
            # Reset detection if we've missed too many frames
            left_line.detected = False
            right_line.detected = False
            result = undist
    
    # Add text to display curvature and offset
    if 'left_curverad' in locals() and 'right_curverad' in locals():
        avg_curve = (left_curverad + right_curverad) / 2
        curve_text = f"Curve Radius: {avg_curve:.2f}m"
        offset_text = f"Vehicle Offset: {abs(vehicle_offset):.2f}m {'right' if vehicle_offset < 0 else 'left'}"
        status_text = f"Lane Tracking: {'Stable' if left_line.miss_count < 3 else 'Recovering...'}"
        
        cv2.putText(result, curve_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, offset_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, status_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result

def main():
    # Use the calibration data from your system
    mtx = np.array([[1.43249747e+03, 0.00000000e+00, 6.75431644e+02],
                    [0.00000000e+00, 1.43065692e+03, 4.57012583e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    dist = np.array([[ 0.01302033, 0.94139111, -0.00436593, 0.00480621, -3.33015441]])
    
    # Initialize camera and lines
    camera = Camera(mtx, dist)
    left_line = Line()
    right_line = Line()
    
    # Setup video capture
    source = input("Enter video source (file path or camera index, default is 0): ") or 0
    try:
        source = int(source)  # Try to convert to int for camera index
    except ValueError:
        pass  # Keep as string for file path
    
    cap = cv2.VideoCapture(source)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties for potential output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Ask if user wants to save video output
    save_output = input("Save video output? (y/n): ").lower() == 'y'
    output_path = None
    
    if save_output:
        output_path = input("Enter output file path (default: output.mp4): ") or "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("End of video stream or error reading frame")
            break
        
        # Process frame for lane detection
        start_time = time.time()
        result = process_frame(frame, left_line, right_line, camera)
        process_time = time.time() - start_time
        
        # Display FPS - add separately to avoid overwriting status text
        fps_text = f"FPS: {1/process_time:.2f}"
        cv2.putText(result, fps_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Lane Detection', result)
        
        # Save frame if required
        if save_output:
            out.write(result)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 