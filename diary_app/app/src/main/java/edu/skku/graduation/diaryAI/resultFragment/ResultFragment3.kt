package edu.skku.graduation.diaryAI.resultFragment

import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.RatingBar
import androidx.fragment.app.viewModels
import androidx.lifecycle.lifecycleScope
import androidx.navigation.NavController
import androidx.navigation.Navigation
import edu.skku.graduation.diaryAI.R
import edu.skku.graduation.diaryAI.manager.DBManager
import edu.skku.graduation.diaryAI.manager.DiaryData
import edu.skku.graduation.diaryAI.manager.ServerManager
import kotlinx.coroutines.launch

class ResultFragment3 : Fragment() {

    private lateinit var navController: NavController

    private var param1: Float = 3F
    private var param2: Float = 3F
    private var param3: Float = 3F
    private var diaryID: Int = 0
    private lateinit var helper: DBManager
    private lateinit var diary: DiaryData

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        helper = DBManager(requireContext(), "diary", null, 1)
        diaryID = requireArguments().getInt("finalID")
        diary = helper.selectDiary(diaryID)
        Log.d("ResultID", diaryID.toString())
        param1 = diary.rating1
        param2 = diary.rating2
        param3 = diary.rating3

        return inflater.inflate(R.layout.fragment_result3, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val r1=view.findViewById<RatingBar>(R.id.ratingBar1)
        val r2=view.findViewById<RatingBar>(R.id.ratingBar2)
        val r3=view.findViewById<RatingBar>(R.id.ratingBar3)

        r1.rating = param1
        r2.rating = param2
        r3.rating = param3

        navController = Navigation.findNavController(view)

        r1.onRatingBarChangeListener =
            RatingBar.OnRatingBarChangeListener { _, rating, _ -> param1 = rating }
        r2.onRatingBarChangeListener =
            RatingBar.OnRatingBarChangeListener { _, rating, _ -> param2 = rating }
        r3.onRatingBarChangeListener =
            RatingBar.OnRatingBarChangeListener { _, rating, _ -> param3 = rating }

        view.findViewById<Button>(R.id.evaluate).setOnClickListener() {
            diary.rating1 = param1
            diary.rating2 = param2
            diary.rating3 = param3
            helper.updateDiary(diary)
            lifecycleScope.launch {
                val server:ServerManager by viewModels()
                server.updateDiaryRequest(diary)
                //성공
                try {
                    val bundle = Bundle()
                    bundle.putInt("clickID", diary.diary_id)
                    navController.navigate(R.id.action_resultFragment3_to_resultFragment2,bundle)
                } catch (e: Exception) {
                    Log.d("result:::::::::::::", "???")
                }
            }
        }
    }
}