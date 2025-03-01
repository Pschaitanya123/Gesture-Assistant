import cv2
import mediapipe as mp
from math import hypot, sin, cos, pi
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import screen_brightness_control as sbc
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import pyautogui
import subprocess
import os

class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Gesture Control System')
        self.setGeometry(300, 300, 500, 450)

        mainLayout = QtWidgets.QVBoxLayout()

        # Control options
        controlBox = QtWidgets.QGroupBox("Control Options")
        controlLayout = QtWidgets.QVBoxLayout()

        self.volumeCheckbox = QtWidgets.QCheckBox('Control Volume (Right Hand)')
        self.volumeCheckbox.setChecked(True)
        self.brightnessCheckbox = QtWidgets.QCheckBox('Control Brightness (Left Hand)')
        self.brightnessCheckbox.setChecked(True)
        self.appDockCheckbox = QtWidgets.QCheckBox('Enable App Dock (Open Hand)')
        self.appDockCheckbox.setChecked(True)

        controlLayout.addWidget(self.volumeCheckbox)
        controlLayout.addWidget(self.brightnessCheckbox)
        controlLayout.addWidget(self.appDockCheckbox)
        controlBox.setLayout(controlLayout)

        # Display options
        displayBox = QtWidgets.QGroupBox("Display Options")
        displayLayout = QtWidgets.QVBoxLayout()

        self.toggleGraphCheckbox = QtWidgets.QCheckBox('Show Hand Landmarks')
        self.toggleGraphCheckbox.setChecked(True)
        self.showDockCheckbox = QtWidgets.QCheckBox('Show App Dock')
        self.showDockCheckbox.setChecked(True)

        displayLayout.addWidget(self.toggleGraphCheckbox)
        displayLayout.addWidget(self.showDockCheckbox)
        displayBox.setLayout(displayLayout)

        # App configuration
        appBox = QtWidgets.QGroupBox("App Dock Configuration")
        appLayout = QtWidgets.QGridLayout()

        # Labels for app names and their respective textboxes
        self.appPaths = []
        self.appNames = ["Chrome", "Edge", "File Explorer", "Notepad", "Calculator", "Settings"]
        
        defaultPaths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"explorer.exe",
            r"notepad.exe",
            r"calc.exe",
            r"ms-settings:"
        ]

        for i, (name, path) in enumerate(zip(self.appNames, defaultPaths)):
            row = i // 2
            col = i % 2 * 2

            label = QtWidgets.QLabel(f"{name}:")
            textbox = QtWidgets.QLineEdit(path)
            self.appPaths.append(textbox)
            
            appLayout.addWidget(label, row, col)
            appLayout.addWidget(textbox, row, col+1)

        appBox.setLayout(appLayout)

        # Status display
        self.statusLabel = QtWidgets.QLabel("Status: Volume: 0% Brightness: 0% App: None")
        self.statusLabel.setAlignment(QtCore.Qt.AlignCenter)

        # Add all widgets to main layout
        mainLayout.addWidget(controlBox)
        mainLayout.addWidget(displayBox)
        mainLayout.addWidget(appBox)
        mainLayout.addWidget(self.statusLabel)

        self.setLayout(mainLayout)
        self.show()

    def updateStatus(self, volume, brightness, app=None):
        status = f"Status: Volume: {volume}% Brightness: {brightness}%"
        if app:
            status += f" App: {app}"
        self.statusLabel.setText(status)

class HandControl:
    def __init__(self, gui):
        self.gui = gui
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Initialize mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Initialize audio control
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.volMin, self.volMax = self.volume.GetVolumeRange()[:2]

        # Initialize brightness control
        self.brightnessMin, self.brightnessMax = 0, 100
        
        # Get initial values
        self.currentVolume = int(np.interp(self.volume.GetMasterVolumeLevel(), 
                                         [self.volMin, self.volMax], 
                                         [0, 100]))
        self.currentBrightness = sbc.get_brightness()[0]
        
        # App dock parameters
        self.app_dock_visible = False
        self.app_dock_active = False
        self.app_dock_radius = 120
        self.app_dock_center = None
        self.selected_app = None
        self.app_launch_cooldown = 0
        self.app_icons = self.load_app_icons()
        
    def load_app_icons(self):
        # Simple colored circles for now, can be replaced with actual icons
        icons = []
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), 
                 (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        
        for color in colors:
            # Create a colored circle icon
            icon = np.zeros((60, 60, 3), dtype=np.uint8)
            cv2.circle(icon, (30, 30), 25, color, -1)
            icons.append(icon)
            
        return icons

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to capture frame. Exiting...")
                break

            # Flip image for more intuitive interaction
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            # Calculate frame height and width
            h, w, c = img.shape
            
            leftHand, rightHand = None, None
            hand_for_dock = None
            
            # Cooldown for app launching to prevent multiple launches
            if self.app_launch_cooldown > 0:
                self.app_launch_cooldown -= 1
            
            if results.multi_hand_landmarks:
                for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, 
                                                    results.multi_handedness)):
                    # Determine hand type
                    if handedness.classification[0].label == "Left":
                        leftHand = hand_landmarks
                    else:
                        rightHand = hand_landmarks
                    
                    # Use the first detected hand for the app dock
                    if hand_idx == 0:
                        hand_for_dock = hand_landmarks

                    # Draw landmarks if enabled
                    if self.gui.toggleGraphCheckbox.isChecked():
                        self.mpDraw.draw_landmarks(img, hand_landmarks, 
                                                 self.mpHands.HAND_CONNECTIONS)

                # Process gestures
                if leftHand and self.gui.brightnessCheckbox.isChecked():
                    self.currentBrightness = self.control_brightness(img, leftHand)
                if rightHand and self.gui.volumeCheckbox.isChecked():
                    self.currentVolume = self.control_volume(img, rightHand)
                
                # Check for app dock activation
                if hand_for_dock and self.gui.appDockCheckbox.isChecked():
                    fingersUp = self.count_fingers_up(hand_for_dock)
                    
                    # If all fingers are up, show the app dock
                    if sum(fingersUp) >= 5:
                        # Get wrist position for dock center
                        wrist = hand_for_dock.landmark[0]
                        cx, cy = int(wrist.x * w), int(wrist.y * h)
                        self.app_dock_center = (cx, cy)
                        self.app_dock_visible = True
                        
                        # Draw the app dock if enabled
                        if self.gui.showDockCheckbox.isChecked():
                            self.draw_app_dock(img)
                            
                        # Check for app selection based on finger extension
                        self.selected_app = self.check_app_selection(hand_for_dock, img)
                        
                        if self.selected_app is not None and self.app_launch_cooldown == 0:
                            self.launch_app(self.selected_app)
                            self.app_launch_cooldown = 30  # Set cooldown to prevent multiple launches
                    else:
                        self.app_dock_visible = False
                        self.selected_app = None
                else:
                    self.app_dock_visible = False
                    self.selected_app = None
            else:
                self.app_dock_visible = False
                self.selected_app = None

            # Update GUI status
            self.gui.updateStatus(self.currentVolume, self.currentBrightness, 
                                  self.selected_app if self.selected_app else None)

            # Display window
            cv2.imshow('Gesture Control', img)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def control_brightness(self, img, hand_landmarks):
        # Get thumb and index finger positions
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Calculate distance
        length = hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        
        # Convert to brightness value
        brightness = np.interp(length, [0.02, 0.2], [self.brightnessMin, self.brightnessMax])
        brightness = int(brightness)
        
        # Set brightness
        sbc.set_brightness(brightness)
        
        # Draw visual feedback
        cv2.putText(img, f'Brightness: {brightness}%', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
        return brightness

    def control_volume(self, img, hand_landmarks):
        # Get thumb and index finger positions
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Calculate distance
        length = hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        
        # Convert to volume value
        vol = np.interp(length, [0.02, 0.2], [self.volMin, self.volMax])
        
        # Set volume
        self.volume.SetMasterVolumeLevel(vol, None)
        
        # Calculate volume percentage
        vol_percentage = int(np.interp(vol, [self.volMin, self.volMax], [0, 100]))
        
        # Draw visual feedback
        cv2.putText(img, f'Volume: {vol_percentage}%', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        return vol_percentage
    
    def count_fingers_up(self, hand_landmarks):
        """Count the number of fingers that are up"""
        tips = [4, 8, 12, 16, 20]  # Landmark indices for fingertips
        lms = hand_landmarks.landmark
        
        # Check if fingers are up
        fingers = []
        
        # Thumb (special case)
        if lms[tips[0]].x < lms[tips[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers
        for tip in tips[1:]:
            if lms[tip].y < lms[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
    
    def draw_app_dock(self, img):
        """Draw the circular app dock"""
        if not self.app_dock_center:
            return
        
        h, w, c = img.shape
        cx, cy = self.app_dock_center
        r = self.app_dock_radius
        
        # Draw main circle
        cv2.circle(img, (cx, cy), r, (200, 200, 200), 2)
        
        # Draw app icons in a circle
        num_apps = len(self.gui.appNames)
        for i in range(num_apps):
            angle = 2 * pi * i / num_apps
            x = int(cx + r * cos(angle))
            y = int(cy + r * sin(angle))
            
            # Calculate icon position
            icon_size = 30
            icon_x = x - icon_size
            icon_y = y - icon_size
            
            # Draw icon background
            cv2.circle(img, (x, y), 25, (50, 50, 50), -1)
            
            # Resize and overlay icon
            resized_icon = cv2.resize(self.app_icons[i], (50, 50))
            
            # Ensure the icon fits within the image boundaries
            if (x-25 >= 0 and y-25 >= 0 and 
                x+25 < w and y+25 < h):
                
                # Create a circular mask
                mask = np.zeros((50, 50), dtype=np.uint8)
                cv2.circle(mask, (25, 25), 25, 255, -1)
                
                # Region of interest in the image
                roi = img[y-25:y+25, x-25:x+25]
                
                # Copy only where the mask is
                for c in range(3):
                    roi[:, :, c] = roi[:, :, c] * (1 - mask/255) + resized_icon[:, :, c] * (mask/255)
            
            # Draw app name
            cv2.putText(img, self.gui.appNames[i], (x-25, y+40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def check_app_selection(self, hand_landmarks, img):
        """Check if a finger is selecting an app"""
        if not self.app_dock_center:
            return None
            
        h, w, c = img.shape
        cx, cy = self.app_dock_center
        r = self.app_dock_radius
        
        # Get fingertip positions
        fingertips = [4, 8, 12, 16, 20]  # Landmark indices for fingertips
        
        for i, tip_idx in enumerate(fingertips):
            tip = hand_landmarks.landmark[tip_idx]
            tip_x, tip_y = int(tip.x * w), int(tip.y * h)
            
            # Get the base of the finger (second knuckle)
            base_idx = tip_idx - 2
            base = hand_landmarks.landmark[base_idx]
            base_x, base_y = int(base.x * w), int(base.y * h)
            
            # Calculate the finger extension vector
            vec_x = tip_x - base_x
            vec_y = tip_y - base_y
            finger_length = hypot(vec_x, vec_y)
            
            # Normalize the vector
            if finger_length > 0:
                vec_x /= finger_length
                vec_y /= finger_length
            
            # Project the vector forward to check for app selection
            projection_length = 1.5 * finger_length  # Make projection longer than finger
            proj_x = base_x + int(vec_x * projection_length)
            proj_y = base_y + int(vec_y * projection_length)
            
            # Draw the projection
            cv2.line(img, (base_x, base_y), (proj_x, proj_y), (0, 255, 255), 2)
            
            # Check if projection intersects with any app icon
            num_apps = len(self.gui.appNames)
            for app_idx in range(num_apps):
                angle = 2 * pi * app_idx / num_apps
                app_x = int(cx + r * cos(angle))
                app_y = int(cy + r * sin(angle))
                
                # Calculate distance from projection endpoint to app icon center
                dist = hypot(proj_x - app_x, proj_y - app_y)
                
                # If close enough, consider it selected
                if dist < 30:
                    # Highlight the selected app
                    cv2.circle(img, (app_x, app_y), 30, (0, 255, 0), 2)
                    return self.gui.appNames[app_idx]
        
        return None
    
    def launch_app(self, app_name):
        """Launch the selected application"""
        try:
            # Find the index of the app in the names list
            idx = self.gui.appNames.index(app_name)
            
            # Get the path from the GUI
            app_path = self.gui.appPaths[idx].text()
            
            # Launch the application
            subprocess.Popen(app_path)
            print(f"Launching: {app_name} ({app_path})")
            
        except Exception as e:
            print(f"Error launching app {app_name}: {e}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = ControlWindow()
    hand_control = HandControl(gui)
    hand_control.run()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()