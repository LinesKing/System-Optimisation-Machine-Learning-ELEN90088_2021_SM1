?	??/?$????/?$??!??/?$??	U???"8@U???"8@!U???"8@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??/?$??)\???(??A???(\???Y?K7?A`??*	     ?g@2U
Iterator::Model::ParallelMapV2R???Q??!      I@)R???Q??1      I@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaph??|?5??!?P^Cy?@)h??|?5??1?P^Cy?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???Q???!Cy?5?@)???Q???1Cy?5?@:Preprocessing2F
Iterator::Model??C?l???!??k(?L@)y?&1???1??P^Cy@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!??P^Cy@)y?&1?|?1??P^Cy@:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 24.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t15.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9U???"8@I+???}?R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	)\???(??)\???(??!)\???(??      ??!       "      ??!       *      ??!       2	???(\??????(\???!???(\???:      ??!       B      ??!       J	?K7?A`???K7?A`??!?K7?A`??R      ??!       Z	?K7?A`???K7?A`??!?K7?A`??b      ??!       JCPU_ONLYYU???"8@b q+???}?R@Y      Y@q??D??c@"?
host?Your program is HIGHLY input-bound because 24.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t15.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQ2"CPU: B 