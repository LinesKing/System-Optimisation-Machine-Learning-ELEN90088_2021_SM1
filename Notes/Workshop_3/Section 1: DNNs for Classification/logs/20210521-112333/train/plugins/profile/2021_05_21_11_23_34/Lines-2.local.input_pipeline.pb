	#??~j???#??~j???!#??~j???	??vR?o?@??vR?o?@!??vR?o?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$#??~j????? ?rh??A? ?rh???Y??v????*	     ?q@2F
Iterator::Model?C?l????!      R@)?? ?rh??1?$I?$IH@:Preprocessing2U
Iterator::Model::ParallelMapV2???x?&??!ܶm۶?7@)???x?&??1ܶm۶?7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??v????!%I?$I?2@)9??v????1%I?$I?2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat{?G?z??!$I?$I?@)?? ?rh??1?$I?$I@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zt?!$I?$I???){?G?zt?1$I?$I???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!I?$I?$??)?~j?t?h?1I?$I?$??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 31.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s8.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??vR?o?@I\b+$Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?? ?rh???? ?rh??!?? ?rh??      ??!       "      ??!       *      ??!       2	? ?rh???? ?rh???!? ?rh???:      ??!       B      ??!       J	??v??????v????!??v????R      ??!       Z	??v??????v????!??v????b      ??!       JCPU_ONLYY??vR?o?@b q\b+$Q@