package edu.skku.graduation.diaryAI.resultFragment

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context.CLIPBOARD_SERVICE
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.RatingBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.fragment.app.Fragment
import androidx.navigation.NavController
import androidx.navigation.Navigation
import edu.skku.graduation.diaryAI.R
import edu.skku.graduation.diaryAI.manager.DBManager
import edu.skku.graduation.diaryAI.manager.DiaryData
import java.text.SimpleDateFormat


class ResultFragment2 : Fragment() {

    private lateinit var navController: NavController
    private var diaryID: Int = 0
    private lateinit var helper: DBManager
    private lateinit var result: DiaryData
    private var rate: Float = 0.0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requireActivity().onBackPressedDispatcher.addCallback(this) {
            navController = Navigation.findNavController(requireView())
            navController.navigate(R.id.action_resultFragment2_to_resultFragment1)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        diaryID = requireArguments().getInt("clickID")
        helper = DBManager(requireContext(), "diary", null, 1)
        result = helper.selectDiary(diaryID)
        rate = (result.rating1 + result.rating2 + result.rating3) / 3
        Log.d("result:::::::::::", result.toString())
        Log.d("RATE:::::::", rate.toString())

        return inflater.inflate(R.layout.fragment_result2, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        navController = Navigation.findNavController(view)
        view.findViewById<Button>(R.id.clipboard).setOnClickListener() {
            val clipboard: ClipboardManager =
                requireActivity().getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("label", view.findViewById<TextView>(R.id.result).text)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(activity, "복사되었습니다", Toast.LENGTH_SHORT).show()
        }

        view.findViewById<Button>(R.id.evaluate).setOnClickListener() {
            val bundle = Bundle()
            bundle.putInt("finalID", diaryID)
            navController.navigate(R.id.action_resultFragment2_to_resultFragment3, bundle)
        }
    }

    override fun onStart() {
        super.onStart()
        view?.findViewById<TextView>(R.id.title)?.text = result.title
        view?.findViewById<TextView>(R.id.content)?.text = result.content
        diaryID = result.diary_id
        view?.findViewById<TextView>(R.id.date)?.text =
            SimpleDateFormat("yy/MM/dd").format(result.date)
        view?.findViewById<RatingBar>(R.id.ratingBar)?.rating = rate
    }
}