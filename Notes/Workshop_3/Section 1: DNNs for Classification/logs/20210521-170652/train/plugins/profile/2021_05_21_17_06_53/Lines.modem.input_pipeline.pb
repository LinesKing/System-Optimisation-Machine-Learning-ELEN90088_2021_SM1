	^?I+??^?I+??!^?I+??	?I?o0@?I?o0@!?I?o0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$^?I+??V-???A/?$???YX9??v??*	     ?h@2U
Iterator::Model::ParallelMapV2??~j?t??!Y?CcC@)??~j?t??1Y?CcC@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapT㥛? ??!%I?$I@@)T㥛? ??1%I?$I@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatL7?A`???!???>4?0@)V-???1?????-@:Preprocessing2F
Iterator::Model?????K??!n۶m?6G@)???Q???1?????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;?O??n??!?S?r
^@);?O??n??1?S?r
^@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!????S @)????Mbp?1????S @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 16.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s8.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?I?o0@I?????T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	V-???V-???!V-???      ??!       "      ??!       *      ??!       2	/?$???/?$???!/?$???:      ??!       B      ??!       J	X9??v??X9??v??!X9??v??R      ??!       Z	X9??v??X9??v??!X9??v??b      ??!       JCPU_ONLYY?I?o0@b q?????T@