?	9??v????9??v????!9??v????	?3?m!^'@?3?m!^'@!?3?m!^'@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$9??v??????~j?t??A?x?&1??Y?? ?rh??*	     ?Y@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Q?????!((((((A@)X9??v???1dddddd>@:Preprocessing2U
Iterator::Model::ParallelMapV2{?G?z??!??????3@){?G?z??1??????3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???S㥛?!yxxxxx:@){?G?z??1??????3@:Preprocessing2F
Iterator::ModelV-???!nnnnnn<@);?O??n??1??????!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip㥛? ???!ddddd?Q@)y?&1?|?1tsssss@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!tsssss@)y?&1?|?1tsssss@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapX9??v???!dddddd>@)????Mbp?1______@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!______@)????Mbp?1______@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t13.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?3?m!^'@I?A?;V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??~j?t????~j?t??!??~j?t??      ??!       "      ??!       *      ??!       2	?x?&1???x?&1??!?x?&1??:      ??!       B      ??!       J	?? ?rh???? ?rh??!?? ?rh??R      ??!       Z	?? ?rh???? ?rh??!?? ?rh??b      ??!       JCPU_ONLYY?3?m!^'@b q?A?;V@Y      Y@q??RJ)?N@"?

both?Your program is MODERATELY input-bound because 11.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nomoderate"t13.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQb?61.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 