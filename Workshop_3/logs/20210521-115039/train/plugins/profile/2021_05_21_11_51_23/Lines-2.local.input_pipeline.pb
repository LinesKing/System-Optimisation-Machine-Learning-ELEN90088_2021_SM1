	?ʡE?????ʡE????!?ʡE????	??L??2(@??L??2(@!??L??2(@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?ʡE????#??~j???ANbX9???Ysh??|???*	     ?f@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?z?G???!??؉??L@)??MbX??1*?2*?2K@:Preprocessing2F
Iterator::ModelJ+???!?N??N?:@)9??v????1$I?$I?,@:Preprocessing2U
Iterator::Model::ParallelMapV2Zd;?O???!eTFeTF)@)Zd;?O???1eTFeTF)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??~j?t??!??$@)?~j?t???1??_??_@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!O??N??@)y?&1?|?1O??N??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?/?$??!N??N?DR@)?~j?t?x?1??_??_
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?x?!??_??_
@)?~j?t?x?1??_??_
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?I+???!?-?-(@)?~j?t?h?1??_??_??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 12.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t11.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??L??2(@Id???U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	#??~j???#??~j???!#??~j???      ??!       "      ??!       *      ??!       2	NbX9???NbX9???!NbX9???:      ??!       B      ??!       J	sh??|???sh??|???!sh??|???R      ??!       Z	sh??|???sh??|???!sh??|???b      ??!       JCPU_ONLYY??L??2(@b qd???U@