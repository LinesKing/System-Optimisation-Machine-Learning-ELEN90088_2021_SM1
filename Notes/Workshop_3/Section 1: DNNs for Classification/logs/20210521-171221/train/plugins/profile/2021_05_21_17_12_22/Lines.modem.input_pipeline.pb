	???Q??????Q???!???Q???	%Zas ?6@%Zas ?6@!%Zas ?6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???Q????&1???A%??C???Y#??~j???*	      q@2U
Iterator::Model::ParallelMapV21?Zd??!KKKKK?C@)1?Zd??1KKKKK?C@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{?G?z??!jiiiii=@){?G?z??1jiiiii=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatJ+???!?????2@)Zd;?O???1jiiii?0@:Preprocessing2F
Iterator::ModelH?z?G??!??????H@)y?&1???1??????$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mb??!??????@)????Mb??1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!????????)?~j?t?h?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 22.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t14.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9$Zas ?6@Iw?'??KS@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?&1????&1???!?&1???      ??!       "      ??!       *      ??!       2	%??C???%??C???!%??C???:      ??!       B      ??!       J	#??~j???#??~j???!#??~j???R      ??!       Z	#??~j???#??~j???!#??~j???b      ??!       JCPU_ONLYY$Zas ?6@b qw?'??KS@