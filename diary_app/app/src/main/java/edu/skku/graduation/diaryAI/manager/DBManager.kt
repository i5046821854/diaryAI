package edu.skku.graduation.diaryAI.manager

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

data class DiaryData(
    var diary_id: Int,
    var title: String,
    var content: String,
    var rating1: Float,
    var rating2: Float,
    var rating3: Float,
    var date: Long
) {}

class DBManager(
    context: Context?,
    name: String?,
    factory: SQLiteDatabase.CursorFactory?,
    version: Int
) : SQLiteOpenHelper(context, name, factory, version) {

    override fun onCreate(db: SQLiteDatabase) {
        val create: String = "CREATE TABLE if not exists diary (" +
                "diary_id integer primary key," +
                "title TEXT NOT NULL, " +
                "content TEXT NOT NULL, " +
                "datetime bigint NOT NULL," +
                "rate1 TEXT NOT NULL," +
                "rate2 TEXT NOT NULL," +
                "rate3 TEXT NOT NULL);"

        db.execSQL(create)
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        val sql = "DROP TABLE if exists diary"
        db.execSQL(sql)
        onCreate(db)
    }

    //insert 메소드
    fun insertDiary(diary: DiaryData) {
        val values = ContentValues()
        //넘겨줄 컬럼의 매개변수 지정
        values.put("diary_id", diary.diary_id)
        values.put("title", diary.title)
        values.put("content", diary.content)
        values.put("rate1", diary.rating1)
        values.put("rate2", diary.rating2)
        values.put("rate3", diary.rating3)
        values.put("datetime", diary.date)
        //쓰기나 수정이 가능한 데이터베이스 변수
        val wd = writableDatabase
        wd.insert("diary", null, values)
        //사용이 끝나면 반드시 close()를 사용하여 메모리누수 가 되지않도록 합시다.
        wd.close()
    }


    //select 메소드
    fun selectDiary(): MutableList<DiaryData> {
        val list = mutableListOf<DiaryData>()
        //전체조회
        val selectAll = "select * from diary"
        //읽기전용 데이터베이스 변수
        val rd = readableDatabase
        //데이터를 받아 줍니다.
        val cursor = rd.rawQuery(selectAll, null)

        //반복문을 사용하여 list 에 데이터를 넘겨 줍시다.
        while (cursor.moveToNext()) {
            val id = cursor.getInt(cursor.getColumnIndexOrThrow("diary_id"))
            val title = cursor.getString(cursor.getColumnIndexOrThrow("title"))
            val content = cursor.getString(cursor.getColumnIndexOrThrow("content"))
            val rate1 = cursor.getFloat(cursor.getColumnIndexOrThrow("rate1"))
            val rate2 = cursor.getFloat(cursor.getColumnIndexOrThrow("rate2"))
            val rate3 = cursor.getFloat(cursor.getColumnIndexOrThrow("rate3"))
            val datetime = cursor.getLong(cursor.getColumnIndexOrThrow("datetime"))

            list.add(DiaryData(id, title, content, rate1, rate2, rate3, datetime))
        }
        cursor.close()
        rd.close()

        return list
    }

    fun selectDiary(ID: Int): DiaryData {
        var query = DiaryData(1, "null", "null", 3.0f, 3.0f,3.0f,0)
        //ID 조회
        val selectID = "select * from diary where diary_id=$ID"
        //읽기전용 데이터베이스 변수
        val rd = readableDatabase
        //데이터를 받아 줍니다.
        val cursor = rd.rawQuery(selectID, null)
        //반복문을 사용하여 list 에 데이터를 넘겨 줍시다.
        while (cursor.moveToNext()) {

            val id = cursor.getInt(cursor.getColumnIndexOrThrow("diary_id"))
            val title = cursor.getString(cursor.getColumnIndexOrThrow("title"))
            val content = cursor.getString(cursor.getColumnIndexOrThrow("content"))
            val rate1 = cursor.getFloat(cursor.getColumnIndexOrThrow("rate1"))
            val rate2 = cursor.getFloat(cursor.getColumnIndexOrThrow("rate2"))
            val rate3 = cursor.getFloat(cursor.getColumnIndexOrThrow("rate3"))
            val datetime = cursor.getLong(cursor.getColumnIndexOrThrow("datetime"))

            query = DiaryData(id, title, content, rate1, rate2, rate3, datetime)
        }
        cursor.close()
        rd.close()

        return query
    }

    //update 메소드
    fun updateDiary(diary: DiaryData) {
        val values = ContentValues()

        values.put("title", diary.title)
        values.put("content", diary.content)
        values.put("rate1", diary.rating1)
        values.put("rate2", diary.rating2)
        values.put("rate3", diary.rating3)
        values.put("datetime", diary.date)

        val wd = writableDatabase
        wd.update("diary", values, "diary_id=${diary.diary_id}", null)
        wd.close()

    }

    //clear
    fun clearDiary() {
        val delete = "delete from diary"
        val db = writableDatabase
        db.execSQL(delete)
        db.close()
    }
}