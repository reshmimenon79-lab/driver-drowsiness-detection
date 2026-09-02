#!/usr/bin/env python
"""Test camera with different backends, settings, and comprehensive diagnostics"""
import cv2
import time
import sys

print("=" * 70)
print("CAMERA DIAGNOSTIC TEST - ROBUST VERSION")
print("=" * 70)

backends = [
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_V4L2, "V4L2"),
    (cv2.CAP_ANY, "Default/Any"),
]

def test_camera_with_backend(idx, backend, name):
    """Test a specific camera with a specific backend"""
    print(f"\n  🔍 Testing camera {idx} with {name} backend...")
    try:
        cap = cv2.VideoCapture(idx, backend)
        
        if not cap.isOpened():
            print(f"    ❌ Could not open camera")
            try:
                cap.release()
            except:
                pass
            return False
        
        print(f"    ✅ Camera opened")
        
        # Try to get camera properties
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"       Resolution: {width}x{height}, FPS: {fps}")
        except Exception as e:
            print(f"       ⚠️  Could not read all properties: {str(e)[:40]}")
        
        # Wait for camera to initialize
        print(f"       Warming up...")
        time.sleep(1.5)
        
        # Clear buffer
        for _ in range(15):
            try:
                cap.read()
                time.sleep(0.05)
            except:
                pass
        
        # Try reading frames with detailed error reporting
        print(f"       Reading frames...")
        frames_read = 0
        frames_valid = 0
        
        for i in range(10):
            try:
                ret, frame = cap.read()
                
                if not ret:
                    print(f"         Frame {i}: ❌ ret=False")
                    continue
                
                if frame is None:
                    print(f"         Frame {i}: ❌ frame is None")
                    continue
                
                if frame.size == 0:
                    print(f"         Frame {i}: ❌ empty frame")
                    continue
                
                frames_read += 1
                brightness = frame.mean()
                shape = frame.shape
                
                # Check frame validity
                if brightness < 10:
                    print(f"         Frame {i}: ⚠️ too dark (brightness={brightness:.1f})")
                elif brightness > 245:
                    print(f"         Frame {i}: ⚠️ too bright (brightness={brightness:.1f})")
                else:
                    frames_valid += 1
                    print(f"         Frame {i}: ✅ OK (shape={shape}, brightness={brightness:.1f})")
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"         Frame {i}: ❌ Error: {str(e)[:40]}")
        
        print(f"       Summary: {frames_read}/10 frames read, {frames_valid}/10 valid")
        
        try:
            cap.release()
            print(f"    ✅ Camera {idx} works with {name}!")
            return True
        except Exception as e:
            print(f"    ⚠️  Error releasing: {str(e)[:40]}")
            return True
    
    except Exception as e:
        print(f"    ❌ Unexpected error: {str(e)[:60]}")
        try:
            cap.release()
        except:
            pass
        return False

print("\n" + "=" * 70)
print("Testing camera backends...")
print("=" * 70)

found_working = False

for backend, name in backends:
    print(f"\n📊 Testing {name} backend (ID: {backend})...")
    
    for idx in range(3):
        if test_camera_with_backend(idx, backend, name):
            found_working = True
            # Don't break - test all cameras with this backend
    
    if found_working:
        print(f"✅ Found working camera with {name} backend")

print("\n" + "=" * 70)
if found_working:
    print("✅ DIAGNOSTIC COMPLETE - Camera is working!")
    print("   You can now run detect.py")
else:
    print("❌ DIAGNOSTIC COMPLETE - No working camera found!")
    print("\n   Troubleshooting steps:")
    print("   1. Check that camera is physically connected")
    print("   2. Verify camera is not in use by another application")
    print("   3. Try restarting the computer")
    print("   4. Update camera driver")
    print("   5. Check Device Manager for camera devices")
    print("   6. Try running as Administrator")
print("=" * 70)
