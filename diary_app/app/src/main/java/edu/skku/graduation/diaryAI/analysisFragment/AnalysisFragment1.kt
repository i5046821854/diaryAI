package edu.skku.graduation.diaryAI.analysisFragment

import android.app.Application
import android.content.Intent
import android.database.Cursor
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.navigation.NavController
import androidx.navigation.Navigation
import edu.skku.graduation.diaryAI.R
import java.io.File


class AnalysisFragment1 : Fragment() {

    lateinit var navController: NavController
    lateinit var fileUri:Uri

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_analysis1, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navController = Navigation.findNavController(view)

        view.findViewById<Button>(R.id.kakao).setOnClickListener(){
            openApplicationOrMarket("com.kakao.talk")
        }


        view.findViewById<Button>(R.id.file).setOnClickListener(){
            getContent.launch("text/plain")

        }

        view.findViewById<Button>(R.id.prev).setOnClickListener(){
            requireActivity().finish()
        }

        view.findViewById<Button>(R.id.next).setOnClickListener(){
            val bundle =  Bundle()
            bundle.putString("path", fileUri.toString())
            navController.navigate(R.id.action_analysisFragment1_to_analysisFragment2,bundle)
        }
    }

    private val getContent = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            fileUri = uri
        }
        view?.findViewById<TextView>(R.id.filename)!!.text= getNameFromURI(uri)

    }

    private fun getNameFromURI(uri: Uri?): String? {
        val contentResolver = context!!.contentResolver
        val c: Cursor? = uri?.let { contentResolver.query(it, null, null, null, null) }
        c?.moveToFirst()
        return c?.getString(c.getColumnIndex(OpenableColumns.DISPLAY_NAME))
    }

    private fun openApplicationOrMarket(packageName: String) {
        var intent = requireContext().packageManager.getLaunchIntentForPackage(packageName)
        if (intent == null) {
            intent = Intent(Intent.ACTION_VIEW)
            intent.data = Uri.parse("market://details?id=$packageName")
        }

        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        requireContext().startActivity(intent)
    }
}