package edu.skku.graduation.diaryAI

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.viewModels
import androidx.lifecycle.lifecycleScope
import edu.skku.graduation.diaryAI.manager.AccountManager
import edu.skku.graduation.diaryAI.manager.ServerManager
import kotlinx.coroutines.*
import org.json.JSONObject

class LoginActivity : AppCompatActivity() {

    lateinit var userID: String
    lateinit var userPW: String
//    private val server = ServerManager()
    private val server:ServerManager by viewModels()
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)
        AccountManager.token.setString("token","")
        findViewById<EditText>(R.id.user_id).setText("lee")
        findViewById<EditText>(R.id.user_pw).setText("123")

        findViewById<Button>(R.id.sign).setOnClickListener {
            userID = findViewById<EditText>(R.id.user_id).text.toString()
            userPW = findViewById<EditText>(R.id.user_pw).text.toString()

            //서버로 전송
            lifecycleScope.launch {
                val result = server.joinRequest(userID, userPW)
                Log.d("RESULT:::::::::::::", result)
                try{
                    val jsonObject = JSONObject(result)
                    val userId = jsonObject.getString("userId")
                    Toast.makeText(baseContext,"회원 가입이 완료되었습니다.",Toast.LENGTH_SHORT).show()
                }catch (e: Exception){
                    Toast.makeText(baseContext,"중복된 사용자가 존재합니다.",Toast.LENGTH_SHORT).show()
                }finally {
                    findViewById<EditText>(R.id.user_id).setText("")
                    findViewById<EditText>(R.id.user_pw).setText("")
                }
            }
        }

        findViewById<Button>(R.id.login).setOnClickListener {
            userID = findViewById<EditText>(R.id.user_id).text.toString()
            userPW = findViewById<EditText>(R.id.user_pw).text.toString()

            val mainIntent = Intent(this, MainActivity::class.java)
//            서버로 전송
            lifecycleScope.launch {
                val result = server.loginRequest(userID, userPW)
                //성공
                try{
                    val jsonObject = JSONObject(result)
                    val token = jsonObject.getString("token")
                    AccountManager.token.setString("token",token)
                    Log.d("TOKEN:::::::::::::", token)
                    startActivity(mainIntent)
                }catch (e: Exception){
                    val jsonObject = JSONObject(result)
                    val error = jsonObject.getString("error")
                    if(error=="잘못된 비밀번호입니다."){
                        findViewById<EditText>(R.id.user_pw).setText("")
                    }else{
                        findViewById<EditText>(R.id.user_id).setText("")
                        findViewById<EditText>(R.id.user_pw).setText("")
                    }
                    Toast.makeText(baseContext,error,Toast.LENGTH_SHORT).show()
                }

            }
        }
    }

}