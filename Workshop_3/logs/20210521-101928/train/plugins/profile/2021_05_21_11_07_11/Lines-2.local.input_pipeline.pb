	?|?5^????|?5^???!?|?5^???	??}??.@??}??.@!??}??.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?|?5^???
ףp=
??A??K7?A??Y??(\?µ?*	     ?S@2U
Iterator::Model::ParallelMapV2??~j?t??!?JG?H@)??~j?t??1?JG?H@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????Mb??!??td?@4@)????Mb??1??td?@4@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t???!???7a.@)?~j?t???1???7a.@:Preprocessing2F
Iterator::Model
ףp=
??!?D?#{L@)y?&1?|?1q?w??!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!???7a@)?~j?t?x?1???7a@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 15.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t16.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??}??.@I]?I?*U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	
ףp=
??
ףp=
??!
ףp=
??      ??!       "      ??!       *      ??!       2	??K7?A????K7?A??!??K7?A??:      ??!       B      ??!       J	??(\?µ???(\?µ?!??(\?µ?R      ??!       Z	??(\?µ???(\?µ?!??(\?µ?b      ??!       JCPU_ONLYY??}??.@b q]?I?*U@