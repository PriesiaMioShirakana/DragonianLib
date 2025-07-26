#pragma once
#include <stdexcept>
#include <vector>
#include <string>

#include "Layers/Modules/Base.h"

namespace DragonianLib
{
	namespace Util
	{
		using Byte = unsigned char;

		struct NumpyHeader
		{
			Byte magic[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
			Byte majorVersion = 1;
			Byte minorVersion = 0;
			uint16_t headerLength = 118;
		};

#if !_MSC_VER
		int _wfopen_s(FILE** _File, const wchar_t* _Filename, const wchar_t* _Mode)
		{
			char _Path[1024];
			char _Mode[10];
			wcstombs(_Path, _Filename, sizeof(_Path));
			wcstombs(_Mode, _Mode, sizeof(_Mode));
			*_File = fopen(_Path, _Mode);
			return (*_File == nullptr) ? errno : 0;
		}
#endif

		class FileGuard  // NOLINT(cppcoreguidelines-special-member-functions)
		{
		public:
			FileGuard(const std::wstring& _Path, const wchar_t* _Mode)
			{
				if (const auto error = _wfopen_s(&FileHandle, _Path.c_str(), _Mode))
					throw std::runtime_error("Failed to open file, error code: " + std::to_string(error));
			}
			~FileGuard()
			{
				if (FileHandle)
					fclose(FileHandle);
			}

			bool Enabled() const
			{
				return FileHandle != nullptr;
			}

			size_t Read(void* _Buffer, size_t _Size, size_t _Count) const
			{
				if (!FileHandle)
					return 0;
				return fread(_Buffer, _Size, _Count, FileHandle);
			}

			size_t Write(const void* _Buffer, size_t _Size, size_t _Count) const
			{
				if (!FileHandle)
					return 0;
				return fwrite(_Buffer, _Size, _Count, FileHandle);
			}

		private:
			FILE* FileHandle = nullptr;
		};

		size_t GetNumpyTypeAligsize(const std::string& _Type);

		std::pair<std::vector<int64_t>, std::vector<Byte>> LoadNumpyFile(const std::wstring& _Path);

		CudaModules::Module::DictType LoadNumpyFileToDict(
			const std::wstring& _Path
		);

		template <typename T>
		std::string GetNumpyTypeString()
		{
			if constexpr (std::is_same_v<T, float>)
				return "f4";
			else if constexpr (std::is_same_v<T, double>)
				return "f8";
			else if constexpr (std::is_same_v<T, int>)
				return "i4";
			else if constexpr (std::is_same_v<T, unsigned int>)
				return "u4";
			else if constexpr (std::is_same_v<T, short>)
				return "i2";
			else if constexpr (std::is_same_v<T, unsigned short>)
				return "u2";
			else if constexpr (std::is_same_v<T, char>)
				return "i1";
			else if constexpr (std::is_same_v<T, unsigned char>)
				return "u1";
			else
				throw std::exception("Unsupported type");
		}

		template <typename ValueType>
		void SaveNumpyFile(
			const std::wstring& _Path,
			const std::vector<unsigned>& _Shape,
			const std::vector<ValueType>& _Data
		)
		{
			unsigned totalSize = 1;
			for (const auto& dim : _Shape)
				totalSize *= dim;

			if (totalSize != _Data.size())
				throw std::exception("Invalid shape");
			const FileGuard _MyFile(_Path, L"wb");
			if (!_MyFile.Enabled())
				throw std::exception("Failed to open file");
			NumpyHeader Header;
			std::string HeaderStr = "{";
			HeaderStr += "'descr': '<" + GetNumpyTypeString<ValueType>() + "', 'fortran_order': False, 'shape': (";
			for (size_t i = 0; i < _Shape.size(); ++i)
			{
				if (i != 0)
					HeaderStr += ", ";
				HeaderStr += std::to_string(_Shape[i]);
			}
			HeaderStr += "), }\n";
			Header.headerLength = static_cast<uint16_t>(HeaderStr.size());
			if (!_MyFile.Write(&Header, sizeof(NumpyHeader), 1))
				throw std::exception("Failed to write header");
			if (!_MyFile.Write(HeaderStr.c_str(), HeaderStr.size(), 1))
				throw std::exception("Failed to write header data");
			if (_MyFile.Write(_Data.data(), sizeof(ValueType), _Data.size()) != _Data.size())
				throw std::exception("Failed to write data");
		}
	}
}
