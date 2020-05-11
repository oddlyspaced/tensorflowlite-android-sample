package com.oddlyspaced.tensortest

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.drawable.GradientDrawable
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.Surface
import com.oddlyspaced.tensortest.tflite.Classifier
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.support.model.Model
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var classifier: Classifier
    private val PICK_IMAGE = 1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        classifier = Classifier.create(this, Classifier.Device.CPU, 4)

        button.setOnClickListener{
            val intent = Intent()
            intent.type = "image/*"
            intent.action = Intent.ACTION_GET_CONTENT
            startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_IMAGE) {
            val uri = data?.data
            val source = ImageDecoder.createSource(this.contentResolver, uri!!)
            val bmp = ImageDecoder.decodeBitmap(source)
            image.setImageBitmap(bmp)
            output.text = classifier.recognizeImage(bmp.copy(Bitmap.Config.ARGB_8888, true), Surface.ROTATION_0)[0].toString()
        }
    }

}
