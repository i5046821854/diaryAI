package edu.skku.graduation.diaryAI.manager

import android.app.Application
import android.content.Context

class AccountManager:Application() {
    companion object {
        lateinit var token: Tokenizer
    }

    override fun onCreate() {
        token = Tokenizer(applicationContext)
        super.onCreate()
    }
}

class Tokenizer(context: Context) {
    private val token = context.getSharedPreferences("token", Context.MODE_PRIVATE)

    fun getString(key: String, defValue: String) : String {
        return token.getString(key, defValue).toString()
    }

    fun setString(key: String, value: String) {
        token.edit().putString(key, value).apply()
    }
}