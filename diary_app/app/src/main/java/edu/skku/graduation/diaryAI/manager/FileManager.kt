package edu.skku.graduation.diaryAI.manager

import android.content.ContentResolver
import android.net.Uri
import android.util.Log
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody
import okio.BufferedSink
import okio.IOException
import okio.source

class FileManager(
    private val contentResolver: ContentResolver,
    private val contentUri: Uri
) : RequestBody() {

    override fun contentType(): MediaType? {
        return "application/json".toMediaTypeOrNull()
    }

    override fun writeTo(sink: BufferedSink) {
        val inputStream = contentResolver.openInputStream(contentUri)
            ?: throw IOException("Couldn't open content URI for reading: $contentUri")

        inputStream.source().use { source ->
            sink.writeAll(source)
        }
    }
}
