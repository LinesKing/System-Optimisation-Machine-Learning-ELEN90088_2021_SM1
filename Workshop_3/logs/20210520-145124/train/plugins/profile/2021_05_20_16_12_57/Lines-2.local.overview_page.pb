?	Zd;?O???Zd;?O???!Zd;?O???	?醪?/*@?醪?/*@!?醪?/*@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Zd;?O?????ʡE???Au?V??Y{?G?z??*	     ?Z@2Z
#Iterator::Model::ParallelMapV2::Zip?v??/??!?>???J@);?O??n??1"5?x+?@@:Preprocessing2U
Iterator::Model::ParallelMapV2?~j?t???!??V?9?6@)?~j?t???1??V?9?6@:Preprocessing2F
Iterator::Model
ףp=
??!j??V?9E@)/?$???1R????3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;?O??n??!"5?x+?0@){?G?z??1?sHM0?"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mb??!??sHM0@)????Mb??1??sHM0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!??sHM0@)????Mbp?1??sHM0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/?$???!R????3@)?~j?t?h?1??V?9?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 13.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t12.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?醪?/*@I?"???U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ʡE?????ʡE???!??ʡE???      ??!       "      ??!       *      ??!       2	u?V??u?V??!u?V??:      ??!       B      ??!       J	{?G?z??{?G?z??!{?G?z??R      ??!       Z	{?G?z??{?G?z??!{?G?z??b      ??!       JCPU_ONLYY?醪?/*@b q?"???U@Y      Y@qd?9eنL@"?

both?Your program is MODERATELY input-bound because 13.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nomoderate"t12.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQb?57.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 