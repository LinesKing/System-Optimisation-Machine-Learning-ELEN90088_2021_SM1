	?"??~j???"??~j??!?"??~j??	?e???-@?e???-@!?e???-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?"??~j??bX9?ȶ?Au?V??YR???Q??*	     ?e@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?? ?rh??!Ȥx?L?C@)?? ?rh??1Ȥx?L?C@:Preprocessing2U
Iterator::Model::ParallelMapV2?Zd;??!?u?7[?A@)?Zd;??1?u?7[?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9??v????!3)^ ??-@)????????1&?dR?,@:Preprocessing2F
Iterator::Model?l??????!?:???CE@)9??v????13)^ ??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!a???@)?~j?t?x?1a???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?!?@&?d??)????MbP?1?@&?d??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 14.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t13.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?e???-@IC?GU@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	bX9?ȶ?bX9?ȶ?!bX9?ȶ?      ??!       "      ??!       *      ??!       2	u?V??u?V??!u?V??:      ??!       B      ??!       J	R???Q??R???Q??!R???Q??R      ??!       Z	R???Q??R???Q??!R???Q??b      ??!       JCPU_ONLYY?e???-@b qC?GU@