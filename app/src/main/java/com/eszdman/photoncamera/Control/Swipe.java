package com.eszdman.photoncamera.Control;
import android.annotation.SuppressLint;
import android.graphics.RectF;
import android.hardware.camera2.CaptureRequest;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;

import androidx.constraintlayout.widget.ConstraintLayout;

import com.eszdman.photoncamera.R;
import com.eszdman.photoncamera.api.CameraReflectionApi;
import com.eszdman.photoncamera.api.Interface;

public class Swipe {
    private static final String TAG = "Swipe";
    private GestureDetector gestureDetector;
    ConstraintLayout manualmode;
    ImageView ocmanual;
    private static int arrowState;
    Animation slideUp;
    Animation slideDown;
    @SuppressLint("ClickableViewAccessibility")
    public void RunDetection(){
        Log.d(TAG,"SwipeDetection - ON");
        manualmode = Interface.i.mainActivity.findViewById(R.id.manual_mode);
        ocmanual = Interface.i.mainActivity.findViewById(R.id.open_close_manual);
        ocmanual.setOnClickListener((v) -> {
            if (arrowState == 0) {
                SwipeUp();
                Log.d(TAG, "Arrow Clicked:SwipeUp");
            } else {
                SwipeDown();
                Log.d(TAG, "Arrow Clicked:SwipeDown");
            }
        });
        slideUp = AnimationUtils.loadAnimation(Interface.i.mainActivity, R.anim.slide_up);
        slideDown = AnimationUtils.loadAnimation(Interface.i.mainActivity, R.anim.animate_slide_down_exit);
        gestureDetector = new GestureDetector(Interface.i.mainActivity, new GestureDetector.SimpleOnGestureListener() {
            private static final int SWIPE_THRESHOLD = 100;
            private static final int SWIPE_VELOCITY_THRESHOLD = 100;
            @Override
            public boolean onDown(MotionEvent e) {
                return true;
            }

            @Override
            public boolean onSingleTapUp(MotionEvent e) {
                startTouchToFocus(e);
                return false;
            }

            @Override
            public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
                Animation slideUp = AnimationUtils.loadAnimation(Interface.i.mainActivity, R.anim.slide_up);
                Animation slideDown = AnimationUtils.loadAnimation(Interface.i.mainActivity, R.anim.animate_slide_down_exit);
                float diffY = e2.getY() - e1.getY();
                float diffX = e2.getX() - e1.getX();
                if (Math.abs(diffX) > Math.abs(diffY)) {
                    if (Math.abs(diffX) > SWIPE_THRESHOLD && Math.abs(velocityX) > SWIPE_VELOCITY_THRESHOLD) {
                        if (diffX > 0) {
                            Log.d(TAG, "Right");
                            SwipeRight();
                        } else {
                            Log.d(TAG, "Left");
                            SwipeLeft();
                        }
                        return true;
                    }
                } else if (Math.abs(diffY) > SWIPE_THRESHOLD && Math.abs(velocityY) > SWIPE_VELOCITY_THRESHOLD) {
                    if (diffY > 0) {
                        Log.d(TAG, "Bottom");//it swipes from top to bottom
                        SwipeDown();
                    } else {
                        Log.d(TAG, "Top");//it swipes from bottom to top
                        SwipeUp();
                    }
                    return true;
                }
                return false;
            }
        });
        View.OnTouchListener touchListener = (view, motionEvent) -> gestureDetector.onTouchEvent(motionEvent);
        View holder = Interface.i.mainActivity.findViewById(R.id.textureHolder);
        Log.d(TAG,"input:"+holder);
        if(holder != null) holder.setOnTouchListener(touchListener);
    }

    private void startTouchToFocus(MotionEvent event)
    {
        //takes into consideration the top and bottom translation of camera_container(if it has been moved due to different display ratios)
        // for calculation of size of viewfinder RectF.(for touch focus detection)
        ConstraintLayout camera_container = Interface.i.mainActivity.findViewById(R.id.camera_container);
        ConstraintLayout layout_viewfinder = Interface.i.mainActivity.findViewById(R.id.layout_viewfinder);
        RectF viewfinderRect = new RectF(
                layout_viewfinder.getLeft(),//left edge of viewfinder
                camera_container.getY(), //y position of camera_container
                layout_viewfinder.getRight(), //right edge of viewfinder
                layout_viewfinder.getBottom() + camera_container.getY() //bottom edge of viewfinder + y position of camera_container
        );
        // Interface.i.camera.showToast(previewRect.toString()+"\nCurX"+event.getX()+"CurY"+event.getY());
        if (viewfinderRect.contains(event.getX(), event.getY())) {
            float translateX = event.getX() - camera_container.getLeft();
            float translateY = event.getY() - camera_container.getTop();
            Interface.i.touchFocus.processTochToFocus(camera_container, translateX, translateY);
        }
    }

    public void SwipeUp(){
        if(!Interface.i.settings.ManualMode) {
            manualmode.startAnimation(slideUp);
            Interface.i.settings.ManualMode = true;
            ocmanual.animate().rotation(180).setDuration(250).start();
        }
        Interface.i.camera.rebuildPreview();
        manualmode.setVisibility(View.VISIBLE);
        arrowState ^= 1;
    }
    public void SwipeDown(){
        if(Interface.i.settings.ManualMode) {
            manualmode.startAnimation(slideDown);
            ocmanual.animate().rotation(0).setDuration(250).start();
        }
        Interface.i.settings.ManualMode = false;

        CameraReflectionApi.set(Interface.i.camera.mPreviewRequest,CaptureRequest.CONTROL_AE_MODE,Interface.i.settings.aeModeOn);
        CameraReflectionApi.set(Interface.i.camera.mPreviewRequest, CaptureRequest.CONTROL_AF_MODE,Interface.i.settings.afMode);
        Interface.i.camera.rebuildPreview();
        manualmode.setVisibility(View.GONE);
        arrowState ^= 1;
    }
    public void SwipeRight(){

    }
    public void SwipeLeft(){

    }
}
