package org.openscanvision

import android.app.Application
import android.util.Log
import org.openscanvision.core.OpenScanVision

class OpenScanVisionApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        val initialized = OpenScanVision.initialize(this)
        if (!initialized) {
            Log.e("OpenScanVision", "OpenCV initialization failed – check native libraries")
        } else {
            Log.d("OpenScanVision", "OpenCV initialized successfully")
        }
    }
}