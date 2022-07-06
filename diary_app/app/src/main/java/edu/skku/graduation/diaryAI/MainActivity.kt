package edu.skku.graduation.diaryAI

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import edu.skku.graduation.diaryAI.db.DBHelper
import edu.skku.graduation.diaryAI.db.DiaryData
import java.util.*

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        findViewById<Button>(R.id.analysis).setOnClickListener {
            startActivity(Intent(this, AnalysisActivity::class.java))
        }

        findViewById<Button>(R.id.result).setOnClickListener {
            startActivity(Intent(this, ResultActivity::class.java))
        }
    }


}