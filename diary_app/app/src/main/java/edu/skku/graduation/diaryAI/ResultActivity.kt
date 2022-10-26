package edu.skku.graduation.diaryAI

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import edu.skku.graduation.diaryAI.manager.ServerManager
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.util.*

class ResultActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)
    }
}