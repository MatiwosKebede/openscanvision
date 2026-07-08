package org.openscanvision

import android.app.Application
import android.util.Log
import org.opencv.android.OpenCVLoader

class OpenScanVisionApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // ─── OpenCV Native Libraries ──────────────────────────────
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV initialization failed – native libraries not loaded")
        } else {
            Log.d("OpenCV", "OpenCV initialized successfully")
        }
    }
}