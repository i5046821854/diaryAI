package edu.skku.graduation.diaryAI.resultFragment

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import androidx.activity.addCallback
import androidx.activity.viewModels
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.lifecycleScope
import androidx.navigation.NavController
import androidx.navigation.Navigation
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.gson.Gson
import edu.skku.graduation.diaryAI.R
import edu.skku.graduation.diaryAI.adapter.RecyclerAdapter
import edu.skku.graduation.diaryAI.manager.DBManager
import edu.skku.graduation.diaryAI.manager.DiaryData
import edu.skku.graduation.diaryAI.manager.ServerManager
import kotlinx.coroutines.launch
import org.json.JSONArray


class ResultFragment1 : Fragment(){

    private lateinit var navController: NavController
    private lateinit var helper: DBManager
    private var linearLayoutManager: RecyclerView.LayoutManager? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requireActivity().onBackPressedDispatcher.addCallback(this) {
            requireActivity().finish()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_result1, container, false)
        val recyclerView: RecyclerView = view.findViewById(R.id.recyclerDiary)
        val adapter = RecyclerAdapter(container)
        helper = DBManager(requireContext(), "diary", null, 1)

        lifecycleScope.launch {
            helper.clearDiary()
            val server:ServerManager by viewModels()
            val result = server.getDiaryRequest()
            //성공
            try {
                val array = JSONArray(result)

                for (i in 0 until array.length()) {
                    val testModel =
                        Gson().fromJson(array.getJSONObject(i).toString(), DiaryData::class.java)
                    helper.insertDiary(testModel)
                }
            } catch (e: Exception) {
                Log.d("result:::::::::::::", "???")
            }finally {
                adapter.listData.addAll(helper.selectDiary())
                adapter.helper = helper

                linearLayoutManager = LinearLayoutManager(activity)
                recyclerView.layoutManager = linearLayoutManager
                recyclerView.adapter = adapter
            }
        }

        return view
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navController = Navigation.findNavController(view)

        view.findViewById<Button>(R.id.prev).setOnClickListener {
            requireActivity().finish()
        }

        view.findViewById<Button>(R.id.refresh).setOnClickListener {
            navController.navigate(R.id.action_resultFragment1_self)
        }
    }
}
